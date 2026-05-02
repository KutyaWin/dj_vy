from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import deque
from contextlib import asynccontextmanager
import csv
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
import heapq
import inspect
import io
import json
import logging
from pathlib import Path
import random
import re
from time import perf_counter
from urllib.parse import urldefrag, urljoin, urlparse
from uuid import uuid4

import aiohttp
import aiofiles
import aiosqlite
from bs4 import BeautifulSoup


TWOPLACES = Decimal("0.01")
INVESTMENT_ASSET_TYPES = ("stocks", "bonds", "etf")


logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_utc_datetime(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise InvalidOperationError(f"{field_name} must be a datetime instance")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def normalize_crawl_record(data: dict[str, object]) -> dict[str, object]:
    if not isinstance(data, dict):
        raise InvalidOperationError("data must be a dictionary")
    url = str(data.get("url", "")).strip()
    if not url:
        raise InvalidOperationError("data['url'] must be a non-empty string")
    crawled_at = data.get("crawled_at")
    if isinstance(crawled_at, datetime):
        crawled_at_value = _normalize_utc_datetime(crawled_at, "crawled_at").isoformat()
    elif isinstance(crawled_at, str) and crawled_at.strip():
        crawled_at_value = crawled_at.strip()
    else:
        crawled_at_value = _utc_now().isoformat()
    links = data.get("links", [])
    metadata = data.get("metadata", {})
    return {
        "url": url,
        "title": str(data.get("title", "")),
        "text": str(data.get("text", "")),
        "links": list(links) if isinstance(links, (list, tuple, set)) else [],
        "metadata": dict(metadata) if isinstance(metadata, dict) else {},
        "crawled_at": crawled_at_value,
        "status_code": int(data["status_code"]) if isinstance(data.get("status_code"), int) else None,
        "content_type": str(data.get("content_type", "")),
    }


class AccountStatus(Enum):
    ACTIVE = "active"
    FROZEN = "frozen"
    CLOSED = "closed"


class Currency(Enum):
    RUB = "RUB"
    USD = "USD"
    EUR = "EUR"
    KZT = "KZT"
    CNY = "CNY"


class ClientStatus(Enum):
    ACTIVE = "active"
    BLOCKED = "blocked"
    SUSPICIOUS = "suspicious"


class AccountFrozenError(Exception):
    pass


class AccountClosedError(Exception):
    pass


class InvalidOperationError(Exception):
    pass


class InsufficientFundsError(Exception):
    pass


@dataclass
class Owner:
    full_name: str
    email: str
    phone: str

    def __post_init__(self) -> None:
        self.full_name = self._validate_text(self.full_name, "full_name")
        self.email = self._validate_email(self.email)
        self.phone = self._validate_text(self.phone, "phone")

    @staticmethod
    def _validate_text(value: str, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise InvalidOperationError(f"{field_name} must be a non-empty string")
        return value.strip()

    @classmethod
    def _validate_email(cls, value: str) -> str:
        email = cls._validate_text(value, "email")
        if "@" not in email:
            raise InvalidOperationError("email must contain '@'")
        if email.count("@") != 1:
            raise InvalidOperationError("email must contain exactly one '@'")
        local_part, domain = email.split("@", 1)
        if not local_part or not domain:
            raise InvalidOperationError("email must contain both local part and domain")
        if "." not in domain:
            raise InvalidOperationError("email domain must contain '.'")
        return email

    def to_safe_dict(self) -> dict[str, str]:
        return {
            "full_name": self.full_name,
            "email": self._mask_email(),
            "phone": self._mask_phone(),
        }

    def _mask_email(self) -> str:
        local_part, domain = self.email.split("@", 1)
        if len(local_part) <= 1:
            masked_local = "*"
        else:
            masked_local = f"{local_part[0]}{'*' * (len(local_part) - 1)}"
        return f"{masked_local}@{domain}"

    def _mask_phone(self) -> str:
        visible_tail = self.phone[-2:] if len(self.phone) >= 2 else self.phone
        masked_head = "*" * max(len(self.phone) - len(visible_tail), 0)
        return f"{masked_head}{visible_tail}"


@dataclass
class Client:
    full_name: str
    email: str
    phone: str
    age: int
    pin_code: str
    client_id: str = field(default_factory=lambda: uuid4().hex[:8])
    status: ClientStatus = ClientStatus.ACTIVE
    account_ids: list[str] = field(default_factory=list)
    failed_auth_attempts: int = 0
    is_locked: bool = False
    suspicious_activity: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.full_name = Owner._validate_text(self.full_name, "full_name")
        self.email = Owner._validate_email(self.email)
        self.phone = Owner._validate_text(self.phone, "phone")
        self.pin_code = self._validate_pin_code(self.pin_code)
        self.client_id = self._validate_client_id(self.client_id)
        self.age = self._validate_age(self.age)
        self.status = self._coerce_status(self.status)

    @staticmethod
    def _validate_age(age: int) -> int:
        if not isinstance(age, int) or isinstance(age, bool):
            raise InvalidOperationError("age must be an integer")
        if age < 18:
            raise InvalidOperationError("client must be at least 18 years old")
        return age

    @staticmethod
    def _validate_pin_code(pin_code: str) -> str:
        if not isinstance(pin_code, str) or not pin_code.isdigit() or len(pin_code) != 4:
            raise InvalidOperationError("pin_code must be a 4-digit string")
        return pin_code

    @staticmethod
    def _validate_client_id(client_id: str) -> str:
        if not isinstance(client_id, str) or not client_id.strip():
            raise InvalidOperationError("client_id must be a non-empty string")
        return client_id.strip()

    @staticmethod
    def _coerce_status(status: ClientStatus | str) -> ClientStatus:
        if isinstance(status, ClientStatus):
            return status
        if isinstance(status, str):
            normalized_status = status.strip().lower()
            for candidate in ClientStatus:
                if candidate.value == normalized_status:
                    return candidate
        raise InvalidOperationError("unsupported client status")

    def add_account_id(self, account_id: str) -> None:
        if account_id not in self.account_ids:
            self.account_ids.append(account_id)

    def add_suspicious_activity(self, event: str) -> None:
        self.suspicious_activity.append(event)
        if not self.is_locked and self.status != ClientStatus.BLOCKED:
            self.status = ClientStatus.SUSPICIOUS

    def to_safe_dict(self) -> dict[str, object]:
        return {
            "client_id": self.client_id,
            "full_name": self.full_name,
            "email": Owner(self.full_name, self.email, self.phone).to_safe_dict()["email"],
            "phone": Owner(self.full_name, self.email, self.phone).to_safe_dict()["phone"],
            "status": self.status.value,
            "account_ids": list(self.account_ids),
        }


class AbstractAccount(ABC):
    def __init__(
        self,
        owner: Owner,
        balance: Decimal | int | float | str = Decimal("0.00"),
        account_id: str | None = None,
        status: AccountStatus | str = AccountStatus.ACTIVE,
        currency: Currency | str = Currency.RUB,
    ) -> None:
        self.owner = self._validate_owner(owner)
        self.account_id = self._validate_account_id(account_id)
        self.status = self._coerce_status(status)
        self.currency = self._coerce_currency(currency)
        self._balance = self._normalize_amount(balance, allow_zero=True)

    @property
    def balance(self) -> Decimal:
        return self._balance

    @abstractmethod
    def deposit(self, amount: Decimal | int | float | str) -> Decimal:
        raise NotImplementedError

    @abstractmethod
    def withdraw(self, amount: Decimal | int | float | str) -> Decimal:
        raise NotImplementedError

    @abstractmethod
    def get_account_info(self) -> dict[str, object]:
        raise NotImplementedError

    @staticmethod
    def _validate_owner(owner: Owner) -> Owner:
        if not isinstance(owner, Owner):
            raise InvalidOperationError("owner must be an Owner instance")
        return owner

    @staticmethod
    def _validate_account_id(account_id: str | None) -> str:
        if account_id is None:
            return uuid4().hex[:8]
        if not isinstance(account_id, str) or not account_id.strip():
            raise InvalidOperationError("account_id must be a non-empty string")
        return account_id.strip()

    @staticmethod
    def _coerce_status(status: AccountStatus | str) -> AccountStatus:
        if isinstance(status, AccountStatus):
            return status
        if isinstance(status, str):
            normalized_status = status.strip().lower()
            for candidate in AccountStatus:
                if candidate.value == normalized_status:
                    return candidate
        raise InvalidOperationError("unsupported account status")

    @staticmethod
    def _coerce_currency(currency: Currency | str) -> Currency:
        if isinstance(currency, Currency):
            return currency
        if isinstance(currency, str):
            normalized_currency = currency.strip().upper()
            for candidate in Currency:
                if candidate.value == normalized_currency:
                    return candidate
        raise InvalidOperationError("unsupported currency")

    @staticmethod
    def _normalize_amount(
        amount: Decimal | int | float | str,
        allow_zero: bool = False,
    ) -> Decimal:
        try:
            normalized_amount = Decimal(str(amount)).quantize(TWOPLACES, rounding=ROUND_HALF_UP)
        except (InvalidOperation, ValueError):
            raise InvalidOperationError("amount must be a valid number") from None

        if allow_zero:
            if normalized_amount < 0:
                raise InvalidOperationError("amount cannot be negative")
            return normalized_amount

        if normalized_amount <= 0:
            raise InvalidOperationError("amount must be greater than zero")
        return normalized_amount

    def _ensure_operations_allowed(self) -> None:
        if self.status == AccountStatus.FROZEN:
            raise AccountFrozenError("account is frozen")
        if self.status == AccountStatus.CLOSED:
            raise AccountClosedError("account is closed")

    def _masked_account_id(self) -> str:
        return f"****{self.account_id[-4:]}"

    @staticmethod
    def _normalize_rate(rate: Decimal | int | float | str) -> Decimal:
        normalized_rate = AbstractAccount._normalize_amount(rate, allow_zero=True)
        return normalized_rate.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


class BankAccount(AbstractAccount):
    def deposit(self, amount: Decimal | int | float | str) -> Decimal:
        self._ensure_operations_allowed()
        normalized_amount = self._normalize_amount(amount)
        self._balance = (self._balance + normalized_amount).quantize(TWOPLACES, rounding=ROUND_HALF_UP)
        return self._balance

    def withdraw(self, amount: Decimal | int | float | str) -> Decimal:
        self._ensure_operations_allowed()
        normalized_amount = self._normalize_amount(amount)
        if normalized_amount > self._balance:
            raise InsufficientFundsError("insufficient funds")
        self._balance = (self._balance - normalized_amount).quantize(TWOPLACES, rounding=ROUND_HALF_UP)
        return self._balance

    def freeze(self) -> AccountStatus:
        if self.status == AccountStatus.CLOSED:
            raise AccountClosedError("closed account cannot be frozen")
        if self.status == AccountStatus.FROZEN:
            raise InvalidOperationError("account is already frozen")
        self.status = AccountStatus.FROZEN
        return self.status

    def activate(self) -> AccountStatus:
        if self.status == AccountStatus.CLOSED:
            raise AccountClosedError("closed account cannot be activated")
        if self.status == AccountStatus.ACTIVE:
            raise InvalidOperationError("account is already active")
        self.status = AccountStatus.ACTIVE
        return self.status

    def close(self) -> AccountStatus:
        if self.status == AccountStatus.CLOSED:
            raise InvalidOperationError("account is already closed")
        self.status = AccountStatus.CLOSED
        return self.status

    def get_account_info(self) -> dict[str, object]:
        return {
            "account_type": self.__class__.__name__,
            "account_id": self._masked_account_id(),
            "owner": self.owner.to_safe_dict(),
            "status": self.status.value,
            "balance": f"{self._balance:.2f}",
            "currency": self.currency.value,
        }

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.owner.full_name}, "
            f"{self._masked_account_id()}, "
            f"{self.status.value}, "
            f"{self._balance:.2f} {self.currency.value}"
            f")"
        )


class SavingsAccount(BankAccount):
    def __init__(
        self,
        owner: Owner,
        balance: Decimal | int | float | str = Decimal("0.00"),
        account_id: str | None = None,
        status: AccountStatus | str = AccountStatus.ACTIVE,
        currency: Currency | str = Currency.RUB,
        min_balance: Decimal | int | float | str = Decimal("0.00"),
        monthly_interest_rate: Decimal | int | float | str = Decimal("0.00"),
    ) -> None:
        super().__init__(owner, balance, account_id, status, currency)
        self.min_balance = self._normalize_amount(min_balance, allow_zero=True)
        self.monthly_interest_rate = self._normalize_rate(monthly_interest_rate)

    def withdraw(self, amount: Decimal | int | float | str) -> Decimal:
        self._ensure_operations_allowed()
        normalized_amount = self._normalize_amount(amount)
        projected_balance = (self._balance - normalized_amount).quantize(TWOPLACES, rounding=ROUND_HALF_UP)
        if projected_balance < self.min_balance:
            raise InvalidOperationError("minimum balance requirement would be violated")
        self._balance = projected_balance
        return self._balance

    def apply_monthly_interest(self) -> Decimal:
        self._ensure_operations_allowed()
        interest = (self._balance * self.monthly_interest_rate).quantize(TWOPLACES, rounding=ROUND_HALF_UP)
        self._balance = (self._balance + interest).quantize(TWOPLACES, rounding=ROUND_HALF_UP)
        return self._balance

    def get_account_info(self) -> dict[str, object]:
        info = super().get_account_info()
        info.update(
            {
                "min_balance": f"{self.min_balance:.2f}",
                "monthly_interest_rate": f"{(self.monthly_interest_rate * Decimal('100')):.2f}%",
            }
        )
        return info

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.owner.full_name}, "
            f"{self._masked_account_id()}, "
            f"{self.status.value}, "
            f"{self._balance:.2f} {self.currency.value}, "
            f"min_balance={self.min_balance:.2f}, "
            f"monthly_interest={(self.monthly_interest_rate * Decimal('100')):.2f}%"
            f")"
        )


class PremiumAccount(BankAccount):
    def __init__(
        self,
        owner: Owner,
        balance: Decimal | int | float | str = Decimal("0.00"),
        account_id: str | None = None,
        status: AccountStatus | str = AccountStatus.ACTIVE,
        currency: Currency | str = Currency.RUB,
        overdraft_limit: Decimal | int | float | str = Decimal("0.00"),
        fixed_commission: Decimal | int | float | str = Decimal("0.00"),
        single_withdrawal_limit: Decimal | int | float | str = Decimal("500000.00"),
    ) -> None:
        super().__init__(owner, balance, account_id, status, currency)
        self.overdraft_limit = self._normalize_amount(overdraft_limit, allow_zero=True)
        self.fixed_commission = self._normalize_amount(fixed_commission, allow_zero=True)
        self.single_withdrawal_limit = self._normalize_amount(single_withdrawal_limit, allow_zero=True)

    def withdraw(self, amount: Decimal | int | float | str) -> Decimal:
        self._ensure_operations_allowed()
        normalized_amount = self._normalize_amount(amount)
        if normalized_amount > self.single_withdrawal_limit:
            raise InvalidOperationError("single withdrawal limit exceeded")
        total_charge = (normalized_amount + self.fixed_commission).quantize(TWOPLACES, rounding=ROUND_HALF_UP)
        allowed_min_balance = (Decimal("0.00") - self.overdraft_limit).quantize(TWOPLACES, rounding=ROUND_HALF_UP)
        projected_balance = (self._balance - total_charge).quantize(TWOPLACES, rounding=ROUND_HALF_UP)
        if projected_balance < allowed_min_balance:
            raise InsufficientFundsError("overdraft limit exceeded")
        self._balance = projected_balance
        return self._balance

    def get_account_info(self) -> dict[str, object]:
        info = super().get_account_info()
        info.update(
            {
                "overdraft_limit": f"{self.overdraft_limit:.2f}",
                "fixed_commission": f"{self.fixed_commission:.2f}",
                "single_withdrawal_limit": f"{self.single_withdrawal_limit:.2f}",
            }
        )
        return info

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.owner.full_name}, "
            f"{self._masked_account_id()}, "
            f"{self.status.value}, "
            f"{self._balance:.2f} {self.currency.value}, "
            f"overdraft={self.overdraft_limit:.2f}, "
            f"commission={self.fixed_commission:.2f}"
            f")"
        )


class InvestmentAccount(BankAccount):
    def __init__(
        self,
        owner: Owner,
        balance: Decimal | int | float | str = Decimal("0.00"),
        account_id: str | None = None,
        status: AccountStatus | str = AccountStatus.ACTIVE,
        currency: Currency | str = Currency.RUB,
        portfolio: dict[str, Decimal | int | float | str] | None = None,
    ) -> None:
        super().__init__(owner, balance, account_id, status, currency)
        self.portfolio = self._normalize_portfolio(portfolio)

    def _normalize_portfolio(
        self,
        portfolio: dict[str, Decimal | int | float | str] | None,
    ) -> dict[str, Decimal]:
        normalized_portfolio = {asset_type: Decimal("0.00") for asset_type in INVESTMENT_ASSET_TYPES}
        if portfolio is None:
            return normalized_portfolio
        if not isinstance(portfolio, dict):
            raise InvalidOperationError("portfolio must be a dictionary")
        for asset_type, asset_value in portfolio.items():
            if asset_type not in INVESTMENT_ASSET_TYPES:
                raise InvalidOperationError("unsupported portfolio asset type")
            normalized_portfolio[asset_type] = self._normalize_amount(asset_value, allow_zero=True)
        return normalized_portfolio

    def _portfolio_total_value(self) -> Decimal:
        total_value = sum(self.portfolio.values(), Decimal("0.00"))
        return total_value.quantize(TWOPLACES, rounding=ROUND_HALF_UP)

    def withdraw(self, amount: Decimal | int | float | str) -> Decimal:
        self._ensure_operations_allowed()
        normalized_amount = self._normalize_amount(amount)
        available_total = (self._balance + self._portfolio_total_value()).quantize(TWOPLACES, rounding=ROUND_HALF_UP)
        if normalized_amount > available_total:
            raise InsufficientFundsError("insufficient funds and portfolio value")
        if normalized_amount <= self._balance:
            self._balance = (self._balance - normalized_amount).quantize(TWOPLACES, rounding=ROUND_HALF_UP)
            return self._balance
        remaining_amount = (normalized_amount - self._balance).quantize(TWOPLACES, rounding=ROUND_HALF_UP)
        self._balance = Decimal("0.00")
        for asset_type in INVESTMENT_ASSET_TYPES:
            asset_value = self.portfolio[asset_type]
            if remaining_amount <= 0:
                break
            deduction = min(asset_value, remaining_amount)
            self.portfolio[asset_type] = (asset_value - deduction).quantize(TWOPLACES, rounding=ROUND_HALF_UP)
            remaining_amount = (remaining_amount - deduction).quantize(TWOPLACES, rounding=ROUND_HALF_UP)
        return self._balance

    def project_yearly_growth(self, yearly_growth_rate: Decimal | int | float | str) -> Decimal:
        normalized_rate = self._normalize_rate(yearly_growth_rate)
        projected_value = (self._portfolio_total_value() * (Decimal("1.00") + normalized_rate)).quantize(
            TWOPLACES,
            rounding=ROUND_HALF_UP,
        )
        return projected_value

    def get_account_info(self) -> dict[str, object]:
        info = super().get_account_info()
        info.update(
            {
                "portfolio": {asset_type: f"{asset_value:.2f}" for asset_type, asset_value in self.portfolio.items()},
                "portfolio_total": f"{self._portfolio_total_value():.2f}",
            }
        )
        return info

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.owner.full_name}, "
            f"{self._masked_account_id()}, "
            f"{self.status.value}, "
            f"cash={self._balance:.2f} {self.currency.value}, "
            f"portfolio_total={self._portfolio_total_value():.2f}"
            f")"
        )


class AuditSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class AuditEvent:
    severity: AuditSeverity | str
    event_type: str
    message: str
    client_id: str | None = None
    account_id: str | None = None
    transaction_id: str | None = None
    risk_level: RiskLevel | str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_utc_now)
    event_id: str = field(default_factory=lambda: uuid4().hex[:12])

    def __post_init__(self) -> None:
        self.severity = self._coerce_severity(self.severity)
        self.event_type = Owner._validate_text(self.event_type, "event_type")
        self.message = Owner._validate_text(self.message, "message")
        self.timestamp = _normalize_utc_datetime(self.timestamp, "timestamp")
        self.event_id = Owner._validate_text(self.event_id, "event_id")
        if self.client_id is not None:
            self.client_id = Owner._validate_text(self.client_id, "client_id")
        if self.account_id is not None:
            self.account_id = Owner._validate_text(self.account_id, "account_id")
        if self.transaction_id is not None:
            self.transaction_id = Owner._validate_text(self.transaction_id, "transaction_id")
        if self.risk_level is not None:
            self.risk_level = self._coerce_risk_level(self.risk_level)
        if not isinstance(self.metadata, dict):
            raise InvalidOperationError("metadata must be a dictionary")

    @staticmethod
    def _coerce_severity(severity: AuditSeverity | str) -> AuditSeverity:
        if isinstance(severity, AuditSeverity):
            return severity
        if isinstance(severity, str):
            normalized_severity = severity.strip().lower()
            for candidate in AuditSeverity:
                if candidate.value == normalized_severity:
                    return candidate
        raise InvalidOperationError("unsupported audit severity")

    @staticmethod
    def _coerce_risk_level(risk_level: RiskLevel | str) -> RiskLevel:
        if isinstance(risk_level, RiskLevel):
            return risk_level
        if isinstance(risk_level, str):
            normalized_risk_level = risk_level.strip().lower()
            for candidate in RiskLevel:
                if candidate.value == normalized_risk_level:
                    return candidate
        raise InvalidOperationError("unsupported risk level")

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "event_type": self.event_type,
            "client_id": self.client_id,
            "account_id": self.account_id,
            "transaction_id": self.transaction_id,
            "risk_level": self.risk_level.value if self.risk_level is not None else None,
            "message": self.message,
            "metadata": self.metadata,
        }


class AuditLog:
    def __init__(self, file_path: str | None = None) -> None:
        if file_path is not None and (not isinstance(file_path, str) or not file_path.strip()):
            raise InvalidOperationError("file_path must be a non-empty string")
        self.file_path = file_path.strip() if isinstance(file_path, str) else None
        self._events: list[AuditEvent] = []

    @property
    def events(self) -> list[AuditEvent]:
        return list(self._events)

    def log_event(
        self,
        severity: AuditSeverity | str,
        event_type: str,
        message: str,
        client_id: str | None = None,
        account_id: str | None = None,
        transaction_id: str | None = None,
        risk_level: RiskLevel | str | None = None,
        metadata: dict[str, object] | None = None,
        timestamp: datetime | None = None,
    ) -> AuditEvent:
        event = AuditEvent(
            severity=severity,
            event_type=event_type,
            message=message,
            client_id=client_id,
            account_id=account_id,
            transaction_id=transaction_id,
            risk_level=risk_level,
            metadata=dict(metadata or {}),
            timestamp=_utc_now() if timestamp is None else _normalize_utc_datetime(timestamp, "timestamp"),
        )
        self._events.append(event)
        if self.file_path is not None:
            path = Path(self.file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        return event

    def filter_events(
        self,
        severity: AuditSeverity | str | None = None,
        event_type: str | None = None,
        client_id: str | None = None,
        transaction_id: str | None = None,
        risk_level: RiskLevel | str | None = None,
    ) -> list[AuditEvent]:
        normalized_severity = AuditEvent._coerce_severity(severity) if severity is not None else None
        normalized_event_type = Owner._validate_text(event_type, "event_type") if event_type is not None else None
        normalized_client_id = Owner._validate_text(client_id, "client_id") if client_id is not None else None
        normalized_transaction_id = (
            Owner._validate_text(transaction_id, "transaction_id") if transaction_id is not None else None
        )
        normalized_risk_level = AuditEvent._coerce_risk_level(risk_level) if risk_level is not None else None
        filtered_events = self._events
        if normalized_severity is not None:
            filtered_events = [event for event in filtered_events if event.severity == normalized_severity]
        if normalized_event_type is not None:
            filtered_events = [event for event in filtered_events if event.event_type == normalized_event_type]
        if normalized_client_id is not None:
            filtered_events = [event for event in filtered_events if event.client_id == normalized_client_id]
        if normalized_transaction_id is not None:
            filtered_events = [event for event in filtered_events if event.transaction_id == normalized_transaction_id]
        if normalized_risk_level is not None:
            filtered_events = [event for event in filtered_events if event.risk_level == normalized_risk_level]
        return list(filtered_events)

    def get_suspicious_events(self, min_risk_level: RiskLevel | str = RiskLevel.MEDIUM) -> list[AuditEvent]:
        normalized_min_risk_level = AuditEvent._coerce_risk_level(min_risk_level)
        risk_order = {RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2, RiskLevel.HIGH: 3}
        return [
            event
            for event in self._events
            if event.risk_level is not None and risk_order[event.risk_level] >= risk_order[normalized_min_risk_level]
        ]

    def get_error_statistics(self) -> dict[str, int]:
        statistics: dict[str, int] = {}
        for event in self._events:
            if event.event_type not in {"transaction_failed", "transaction_blocked", "operation_error"}:
                continue
            statistics[event.message] = statistics.get(event.message, 0) + 1
        return statistics


@dataclass
class RiskAssessment:
    risk_level: RiskLevel | str
    reasons: list[str] = field(default_factory=list)
    should_block: bool = False
    score: int = 0

    def __post_init__(self) -> None:
        self.risk_level = AuditEvent._coerce_risk_level(self.risk_level)
        if not isinstance(self.reasons, list) or any(not isinstance(reason, str) or not reason.strip() for reason in self.reasons):
            raise InvalidOperationError("reasons must be a list of non-empty strings")
        if not isinstance(self.should_block, bool):
            raise InvalidOperationError("should_block must be a boolean")
        if not isinstance(self.score, int) or self.score < 0:
            raise InvalidOperationError("score must be a non-negative integer")


class RiskAnalyzer:
    def __init__(
        self,
        large_amount_threshold: Decimal | int | float | str = Decimal("1000.00"),
        frequent_operations_threshold: int = 3,
        frequent_operations_window_minutes: int = 10,
        block_level: RiskLevel | str = RiskLevel.HIGH,
        base_currency: Currency | str = Currency.RUB,
        reference_exchange_rates: dict[tuple[str, str], Decimal | int | float | str] | None = None,
    ) -> None:
        self.large_amount_threshold = Transaction._normalize_amount(large_amount_threshold)
        if not isinstance(frequent_operations_threshold, int) or frequent_operations_threshold <= 0:
            raise InvalidOperationError("frequent_operations_threshold must be a positive integer")
        if not isinstance(frequent_operations_window_minutes, int) or frequent_operations_window_minutes <= 0:
            raise InvalidOperationError("frequent_operations_window_minutes must be a positive integer")
        self.frequent_operations_threshold = frequent_operations_threshold
        self.frequent_operations_window_minutes = frequent_operations_window_minutes
        self.block_level = AuditEvent._coerce_risk_level(block_level)
        self.base_currency = Transaction._coerce_currency(base_currency)
        default_reference_rates: dict[tuple[str, str], Decimal | int | float | str] = {
            (Currency.USD.value, Currency.RUB.value): "90.00",
            (Currency.EUR.value, Currency.RUB.value): "98.00",
            (Currency.KZT.value, Currency.RUB.value): "0.18",
            (Currency.CNY.value, Currency.RUB.value): "12.50",
        }
        normalized_reference_rates: dict[tuple[str, str], Decimal] = {}
        for currency_pair, rate in (reference_exchange_rates or default_reference_rates).items():
            if not isinstance(currency_pair, tuple) or len(currency_pair) != 2:
                raise InvalidOperationError("reference exchange rate key must be a pair of currency codes")
            source_currency = Transaction._coerce_currency(currency_pair[0]).value
            target_currency = Transaction._coerce_currency(currency_pair[1]).value
            normalized_reference_rates[(source_currency, target_currency)] = Transaction._normalize_amount(rate)
        self.reference_exchange_rates = normalized_reference_rates

    def _risk_from_score(self, score: int) -> RiskLevel:
        if score >= 4:
            return RiskLevel.HIGH
        if score >= 2:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _should_block(self, risk_level: RiskLevel) -> bool:
        risk_order = {RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2, RiskLevel.HIGH: 3}
        return risk_order[risk_level] >= risk_order[self.block_level]

    def _normalize_to_base_currency(self, amount: Decimal, currency: Currency) -> Decimal | None:
        if currency == self.base_currency:
            return amount.quantize(TWOPLACES, rounding=ROUND_HALF_UP)
        direct_pair = (currency.value, self.base_currency.value)
        if direct_pair in self.reference_exchange_rates:
            return (amount * self.reference_exchange_rates[direct_pair]).quantize(TWOPLACES, rounding=ROUND_HALF_UP)
        reverse_pair = (self.base_currency.value, currency.value)
        if reverse_pair in self.reference_exchange_rates:
            reverse_rate = self.reference_exchange_rates[reverse_pair]
            return (amount / reverse_rate).quantize(TWOPLACES, rounding=ROUND_HALF_UP)
        return None

    def analyze_transaction(self, transaction: Transaction, bank: Bank, current_time: datetime) -> RiskAssessment:
        if not isinstance(transaction, Transaction):
            raise InvalidOperationError("transaction must be a Transaction instance")
        if not isinstance(bank, Bank):
            raise InvalidOperationError("bank must be a Bank instance")
        normalized_current_time = _normalize_utc_datetime(current_time, "current_time")
        sender_client = bank._get_account_owner_client(transaction.sender_account_id)
        reasons: list[str] = []
        score = 0
        normalized_amount = self._normalize_to_base_currency(transaction.amount, transaction.currency)
        if normalized_amount is not None and normalized_amount >= self.large_amount_threshold:
            reasons.append("large amount")
            score += 2
        recent_events = [
            event
            for event in bank.audit_log.events
            if event.client_id == sender_client.client_id
            and event.event_type in {"transaction_completed", "transaction_failed", "transaction_blocked"}
            and event.timestamp >= normalized_current_time - timedelta(minutes=self.frequent_operations_window_minutes)
        ]
        if len(recent_events) >= self.frequent_operations_threshold:
            reasons.append("frequent operations")
            score += 1
        known_recipient_events = [
            event
            for event in bank.audit_log.events
            if event.client_id == sender_client.client_id
            and event.metadata.get("recipient_account_id") == transaction.recipient_account_id
            and event.event_type in {"transaction_completed", "transaction_blocked", "transaction_failed"}
        ]
        if not known_recipient_events:
            reasons.append("transfer to a new account")
            score += 1
        if 0 <= normalized_current_time.hour < 5:
            reasons.append("night operation")
            score += 2
        risk_level = self._risk_from_score(score)
        return RiskAssessment(
            risk_level=risk_level,
            reasons=reasons,
            should_block=self._should_block(risk_level),
            score=score,
        )


class Bank:
    def __init__(self, name: str = "Bank", audit_log_file_path: str | None = None) -> None:
        self.name = Owner._validate_text(name, "name")
        self.clients: dict[str, Client] = {}
        self.accounts: dict[str, AbstractAccount] = {}
        self.account_to_client: dict[str, str] = {}
        self.audit_log = AuditLog(file_path=audit_log_file_path)
        self.risk_analyzer = RiskAnalyzer()

    @staticmethod
    def _generate_short_id() -> str:
        return uuid4().hex[:8]

    def _generate_unique_account_id(self) -> str:
        account_id = self._generate_short_id()
        while account_id in self.accounts:
            account_id = self._generate_short_id()
        return account_id

    def _get_client(self, client_id: str) -> Client:
        normalized_client_id = Owner._validate_text(client_id, "client_id")
        if normalized_client_id not in self.clients:
            raise InvalidOperationError("client not found")
        return self.clients[normalized_client_id]

    def _get_account(self, account_id: str) -> AbstractAccount:
        normalized_account_id = Owner._validate_text(account_id, "account_id")
        if normalized_account_id not in self.accounts:
            raise InvalidOperationError("account not found")
        return self.accounts[normalized_account_id]

    def _get_account_owner_client(self, account_id: str) -> Client:
        normalized_account_id = Owner._validate_text(account_id, "account_id")
        if normalized_account_id not in self.account_to_client:
            raise InvalidOperationError("account not linked to client")
        return self._get_client(self.account_to_client[normalized_account_id])

    def _mark_suspicious(
        self,
        client: Client,
        event: str,
        risk_level: RiskLevel | str = RiskLevel.MEDIUM,
    ) -> None:
        normalized_risk_level = AuditEvent._coerce_risk_level(risk_level)
        severity_by_risk_level = {
            RiskLevel.LOW: AuditSeverity.LOW,
            RiskLevel.MEDIUM: AuditSeverity.MEDIUM,
            RiskLevel.HIGH: AuditSeverity.HIGH,
        }
        client.add_suspicious_activity(event)
        self.audit_log.log_event(
            severity=severity_by_risk_level[normalized_risk_level],
            event_type="suspicious_activity",
            message=event,
            client_id=client.client_id,
            risk_level=normalized_risk_level,
            metadata={"client_status": client.status.value},
        )

    def log_audit_event(
        self,
        severity: AuditSeverity | str,
        event_type: str,
        message: str,
        client_id: str | None = None,
        account_id: str | None = None,
        transaction_id: str | None = None,
        risk_level: RiskLevel | str | None = None,
        metadata: dict[str, object] | None = None,
        timestamp: datetime | None = None,
    ) -> AuditEvent:
        return self.audit_log.log_event(
            severity=severity,
            event_type=event_type,
            message=message,
            client_id=client_id,
            account_id=account_id,
            transaction_id=transaction_id,
            risk_level=risk_level,
            metadata=metadata,
            timestamp=timestamp,
        )

    def _current_hour(self, current_time: datetime | None = None) -> int:
        if current_time is None:
            return _utc_now().hour
        return _normalize_utc_datetime(current_time, "current_time").hour

    def _ensure_operation_time_allowed(
        self,
        client: Client,
        action_name: str,
        current_time: datetime | None = None,
    ) -> None:
        current_hour = self._current_hour(current_time)
        if 0 <= current_hour < 5:
            self._mark_suspicious(client, f"restricted hours operation: {action_name}", RiskLevel.MEDIUM)
            raise InvalidOperationError("operations are not allowed from 00:00 to 05:00")

    def _ensure_client_is_not_blocked(self, client: Client, action_name: str) -> None:
        if client.is_locked or client.status == ClientStatus.BLOCKED:
            self._mark_suspicious(client, f"blocked client attempted to {action_name}", RiskLevel.HIGH)
            raise InvalidOperationError(f"blocked client cannot {action_name.replace('_', ' ')}")

    def add_client(self, client: Client) -> Client:
        if not isinstance(client, Client):
            raise InvalidOperationError("client must be a Client instance")
        if client.client_id in self.clients:
            raise InvalidOperationError("client_id must be unique")
        self.clients[client.client_id] = client
        self.log_audit_event(
            severity=AuditSeverity.LOW,
            event_type="client_added",
            message="client added to bank",
            client_id=client.client_id,
        )
        return client

    def open_account(self, client_id: str, account_type: str = "bank", **kwargs: object) -> AbstractAccount:
        client = self._get_client(client_id)
        self._ensure_operation_time_allowed(client, "open_account")
        self._ensure_client_is_not_blocked(client, "open_account")
        normalized_account_type = Owner._validate_text(account_type, "account_type").lower()
        account_classes: dict[str, type[AbstractAccount]] = {
            "bank": BankAccount,
            "savings": SavingsAccount,
            "premium": PremiumAccount,
            "investment": InvestmentAccount,
        }
        if normalized_account_type not in account_classes:
            raise InvalidOperationError("unsupported account type")
        owner = Owner(client.full_name, client.email, client.phone)
        account_kwargs = dict(kwargs)
        account_kwargs["owner"] = owner
        provided_account_id = account_kwargs.get("account_id")
        if provided_account_id is None or Owner._validate_text(str(provided_account_id), "account_id") in self.accounts:
            account_kwargs["account_id"] = self._generate_unique_account_id()
        account = account_classes[normalized_account_type](**account_kwargs)
        self.accounts[account.account_id] = account
        self.account_to_client[account.account_id] = client.client_id
        client.add_account_id(account.account_id)
        account_info = account.get_account_info()
        self.log_audit_event(
            severity=AuditSeverity.LOW,
            event_type="account_opened",
            message="account opened",
            client_id=client.client_id,
            account_id=account.account_id,
            metadata={
                "account_type": account_info["account_type"],
                "account_status": account_info["status"],
                "currency": account_info["currency"],
                "balance": account_info["balance"],
                "client_status": client.status.value,
            },
        )
        return account

    def close_account(self, account_id: str) -> AccountStatus:
        account = self._get_account(account_id)
        client = self._get_account_owner_client(account_id)
        self._ensure_operation_time_allowed(client, "close_account")
        self._ensure_client_is_not_blocked(client, "close_account")
        status = account.close()
        self.log_audit_event(
            severity=AuditSeverity.MEDIUM,
            event_type="account_closed",
            message="account closed",
            client_id=client.client_id,
            account_id=account.account_id,
            metadata={
                "account_type": account.__class__.__name__,
                "account_status": status.value,
                "currency": account.currency.value,
                "client_status": client.status.value,
            },
        )
        return status

    def freeze_account(self, account_id: str) -> AccountStatus:
        account = self._get_account(account_id)
        client = self._get_account_owner_client(account_id)
        self._ensure_operation_time_allowed(client, "freeze_account")
        self._ensure_client_is_not_blocked(client, "freeze_account")
        status = account.freeze()
        self.log_audit_event(
            severity=AuditSeverity.HIGH,
            event_type="account_frozen",
            message="account frozen",
            client_id=client.client_id,
            account_id=account.account_id,
            metadata={
                "account_type": account.__class__.__name__,
                "account_status": status.value,
                "currency": account.currency.value,
                "client_status": client.status.value,
            },
        )
        return status

    def unfreeze_account(self, account_id: str) -> AccountStatus:
        account = self._get_account(account_id)
        client = self._get_account_owner_client(account_id)
        self._ensure_operation_time_allowed(client, "unfreeze_account")
        self._ensure_client_is_not_blocked(client, "unfreeze_account")
        status = account.activate()
        self.log_audit_event(
            severity=AuditSeverity.MEDIUM,
            event_type="account_unfrozen",
            message="account unfrozen",
            client_id=client.client_id,
            account_id=account.account_id,
            metadata={
                "account_type": account.__class__.__name__,
                "account_status": status.value,
                "currency": account.currency.value,
                "client_status": client.status.value,
            },
        )
        return status

    def authenticate_client(self, client_id: str, pin_code: str) -> bool:
        client = self._get_client(client_id)
        if client.is_locked:
            self._mark_suspicious(client, "authentication attempt on blocked client", RiskLevel.HIGH)
            raise InvalidOperationError("client is blocked")
        if client.pin_code != pin_code:
            client.failed_auth_attempts += 1
            self.log_audit_event(
                severity=AuditSeverity.MEDIUM,
                event_type="authentication_failed",
                message="invalid credentials",
                client_id=client.client_id,
                risk_level=RiskLevel.MEDIUM,
                metadata={
                    "failed_auth_attempts": client.failed_auth_attempts,
                    "remaining_attempts": max(0, 3 - client.failed_auth_attempts),
                },
            )
            if client.failed_auth_attempts >= 3:
                client.is_locked = True
                client.status = ClientStatus.BLOCKED
                self._mark_suspicious(
                    client,
                    "client locked after 3 failed authentication attempts",
                    RiskLevel.HIGH,
                )
                self.log_audit_event(
                    severity=AuditSeverity.HIGH,
                    event_type="client_blocked",
                    message="client blocked after repeated authentication failures",
                    client_id=client.client_id,
                    risk_level=RiskLevel.HIGH,
                    metadata={"failed_auth_attempts": client.failed_auth_attempts},
                )
            raise InvalidOperationError("invalid credentials")
        client.failed_auth_attempts = 0
        return True

    def search_accounts(
        self,
        client_id: str | None = None,
        status: AccountStatus | str | None = None,
        account_type: str | None = None,
    ) -> list[AbstractAccount]:
        matched_accounts = list(self.accounts.values())
        if client_id is not None:
            client = self._get_client(client_id)
            matched_accounts = [account for account in matched_accounts if account.account_id in client.account_ids]
        if status is not None:
            normalized_status = AbstractAccount._coerce_status(status)
            matched_accounts = [account for account in matched_accounts if account.status == normalized_status]
        if account_type is not None:
            normalized_account_type = Owner._validate_text(account_type, "account_type").lower()
            matched_accounts = [
                account for account in matched_accounts if account.__class__.__name__.lower() == f"{normalized_account_type}account"
            ]
        return matched_accounts

    def _empty_balances_by_currency(self) -> dict[str, Decimal]:
        return {currency.value: Decimal("0.00") for currency in Currency}

    def _account_total_assets(self, account: AbstractAccount) -> Decimal:
        total_assets = account.balance
        if isinstance(account, InvestmentAccount):
            total_assets += account._portfolio_total_value()
        return total_assets.quantize(TWOPLACES, rounding=ROUND_HALF_UP)

    def get_total_balance(self) -> dict[str, Decimal]:
        total_balance = self._empty_balances_by_currency()
        for account in self.accounts.values():
            total_balance[account.currency.value] += self._account_total_assets(account)
        return {
            currency_code: amount.quantize(TWOPLACES, rounding=ROUND_HALF_UP)
            for currency_code, amount in total_balance.items()
        }

    def get_clients_ranking(self) -> dict[str, list[dict[str, object]]]:
        ranking_by_currency: dict[str, list[dict[str, object]]] = {
            currency.value: [] for currency in Currency
        }
        for client in self.clients.values():
            client_assets_by_currency = self._empty_balances_by_currency()
            accounts_count_by_currency = {currency.value: 0 for currency in Currency}
            for account_id in client.account_ids:
                account = self.accounts[account_id]
                currency_code = account.currency.value
                client_assets_by_currency[currency_code] += self._account_total_assets(account)
                accounts_count_by_currency[currency_code] += 1
            for currency_code, total_assets in client_assets_by_currency.items():
                if accounts_count_by_currency[currency_code] == 0:
                    continue
                ranking_by_currency[currency_code].append(
                    {
                        "client_id": client.client_id,
                        "full_name": client.full_name,
                        "status": client.status.value,
                        "accounts_count": accounts_count_by_currency[currency_code],
                        "total_assets": f"{total_assets.quantize(TWOPLACES, rounding=ROUND_HALF_UP):.2f}",
                    }
                )
        for currency_code in ranking_by_currency:
            ranking_by_currency[currency_code].sort(
                key=lambda item: Decimal(str(item["total_assets"])),
                reverse=True,
            )
        return ranking_by_currency

    def get_audit_report_suspicious_operations(
        self,
        min_risk_level: RiskLevel | str = RiskLevel.MEDIUM,
    ) -> list[dict[str, object]]:
        return [event.to_dict() for event in self.audit_log.get_suspicious_events(min_risk_level)]

    def get_client_risk_profile(self, client_id: str) -> dict[str, object]:
        client = self._get_client(client_id)
        client_events = self.audit_log.filter_events(client_id=client.client_id)
        risk_events = [event for event in client_events if event.risk_level is not None]
        blocked_operations = [event for event in client_events if event.event_type == "transaction_blocked"]
        risk_order = {RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2, RiskLevel.HIGH: 3}
        highest_risk = RiskLevel.LOW
        for event in risk_events:
            if event.risk_level is not None and risk_order[event.risk_level] > risk_order[highest_risk]:
                highest_risk = event.risk_level
        recent_risky_recipients = sorted(
            {
                str(event.metadata.get("recipient_account_id"))
                for event in risk_events
                if event.metadata.get("recipient_account_id") is not None
            }
        )
        return {
            "client_id": client.client_id,
            "client_status": client.status.value,
            "highest_risk": highest_risk.value,
            "suspicious_events_count": len([event for event in client_events if event.event_type == "suspicious_activity"]),
            "blocked_operations_count": len(blocked_operations),
            "total_risk_events": len(risk_events),
            "recent_risky_recipients": recent_risky_recipients,
        }

    def get_audit_error_statistics(self) -> dict[str, int]:
        return self.audit_log.get_error_statistics()


class ReportType(Enum):
    CLIENT = "client"
    BANK = "bank"
    RISK = "risk"


class ReportBuilder:
    def __init__(self, bank: Bank) -> None:
        if not isinstance(bank, Bank):
            raise InvalidOperationError("bank must be a Bank instance")
        self.bank = bank

    def _validate_path(self, path_value: str, field_name: str) -> Path:
        if not isinstance(path_value, str) or not path_value.strip():
            raise InvalidOperationError(f"{field_name} must be a non-empty string")
        return Path(path_value.strip())

    def _serialize_for_export(self, value: object) -> object:
        if isinstance(value, Decimal):
            return f"{value.quantize(TWOPLACES, rounding=ROUND_HALF_UP):.2f}"
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {str(key): self._serialize_for_export(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._serialize_for_export(item) for item in value]
        if isinstance(value, tuple):
            return [self._serialize_for_export(item) for item in value]
        return value

    def _flatten_report_data(self, value: object, prefix: str = "") -> list[tuple[str, str]]:
        serialized_value = self._serialize_for_export(value)
        if isinstance(serialized_value, dict):
            flattened_rows: list[tuple[str, str]] = []
            for key, item in serialized_value.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                flattened_rows.extend(self._flatten_report_data(item, next_prefix))
            return flattened_rows
        if isinstance(serialized_value, list):
            flattened_rows = []
            for index, item in enumerate(serialized_value):
                next_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
                flattened_rows.extend(self._flatten_report_data(item, next_prefix))
            return flattened_rows
        return [(prefix or "value", "" if serialized_value is None else str(serialized_value))]

    def _build_client_assets_by_currency(self, client: Client) -> dict[str, str]:
        balances = self.bank._empty_balances_by_currency()
        for account_id in client.account_ids:
            account = self.bank.accounts[account_id]
            balances[account.currency.value] += self.bank._account_total_assets(account)
        return {
            currency_code: f"{amount.quantize(TWOPLACES, rounding=ROUND_HALF_UP):.2f}"
            for currency_code, amount in balances.items()
            if amount > 0
        }

    @staticmethod
    def _event_type_to_transaction_status(event_type: str) -> str:
        if event_type == "transaction_completed":
            return TransactionStatus.COMPLETED.value
        if event_type in {"transaction_failed", "transaction_blocked"}:
            return TransactionStatus.FAILED.value
        if event_type == "risk_assessment":
            return TransactionStatus.PENDING.value
        return event_type

    def _build_transaction_summaries(self) -> list[dict[str, object]]:
        summaries_by_id: dict[str, dict[str, object]] = {}
        for event in self.bank.audit_log.events:
            if event.transaction_id is None:
                continue
            metadata = event.metadata
            sender_account_id = metadata.get("sender_account_id")
            recipient_account_id = metadata.get("recipient_account_id")
            sender_client_id = None
            recipient_client_id = None
            if isinstance(sender_account_id, str) and sender_account_id in self.bank.account_to_client:
                sender_client_id = self.bank.account_to_client[sender_account_id]
            if isinstance(recipient_account_id, str) and recipient_account_id in self.bank.account_to_client:
                recipient_client_id = self.bank.account_to_client[recipient_account_id]
            summary = summaries_by_id.setdefault(
                event.transaction_id,
                {
                    "transaction_id": event.transaction_id,
                    "client_id": sender_client_id or event.client_id,
                    "sender_client_id": sender_client_id,
                    "recipient_client_id": recipient_client_id,
                    "sender_account_id": sender_account_id,
                    "recipient_account_id": recipient_account_id,
                    "transaction_type": metadata.get("transaction_type"),
                    "amount": metadata.get("amount"),
                    "currency": metadata.get("currency"),
                    "risk_level": None,
                    "latest_event_type": event.event_type,
                    "status": self._event_type_to_transaction_status(event.event_type),
                    "message": event.message,
                    "last_updated": event.timestamp.isoformat(),
                },
            )
            summary["latest_event_type"] = event.event_type
            summary["status"] = self._event_type_to_transaction_status(event.event_type)
            summary["message"] = event.message
            summary["last_updated"] = event.timestamp.isoformat()
            if event.risk_level is not None:
                summary["risk_level"] = event.risk_level.value
            if summary["transaction_type"] is None:
                summary["transaction_type"] = metadata.get("transaction_type")
            if summary["amount"] is None:
                summary["amount"] = metadata.get("amount")
            if summary["currency"] is None:
                summary["currency"] = metadata.get("currency")
        return sorted(summaries_by_id.values(), key=lambda item: str(item["last_updated"]))

    def _build_client_status_distribution(self) -> dict[str, int]:
        distribution: dict[str, int] = {}
        for client in self.bank.clients.values():
            distribution[client.status.value] = distribution.get(client.status.value, 0) + 1
        return distribution

    def _build_account_distribution_by_currency(self) -> dict[str, int]:
        distribution: dict[str, int] = {currency.value: 0 for currency in Currency}
        for account in self.bank.accounts.values():
            distribution[account.currency.value] += 1
        return distribution

    def _build_account_distribution_by_type(self) -> dict[str, int]:
        distribution: dict[str, int] = {}
        for account in self.bank.accounts.values():
            account_type = account.__class__.__name__
            distribution[account_type] = distribution.get(account_type, 0) + 1
        return distribution

    def _build_transaction_statistics(self) -> dict[str, object]:
        by_status: dict[str, int] = {}
        by_type: dict[str, int] = {}
        transaction_summaries = self._build_transaction_summaries()
        for summary in transaction_summaries:
            status = str(summary["status"])
            transaction_type = str(summary["transaction_type"])
            by_status[status] = by_status.get(status, 0) + 1
            by_type[transaction_type] = by_type.get(transaction_type, 0) + 1
        return {
            "total": len(transaction_summaries),
            "by_status": by_status,
            "by_type": by_type,
        }

    def _build_risk_level_distribution(self) -> dict[str, int]:
        distribution = {risk_level.value: 0 for risk_level in RiskLevel}
        for event in self.bank.audit_log.events:
            if event.risk_level is None:
                continue
            distribution[event.risk_level.value] += 1
        return distribution

    def _build_top_risky_clients(self, limit: int = 5) -> list[dict[str, object]]:
        profiles: list[dict[str, object]] = []
        risk_order = {RiskLevel.LOW.value: 1, RiskLevel.MEDIUM.value: 2, RiskLevel.HIGH.value: 3}
        for client in self.bank.clients.values():
            profile = self.bank.get_client_risk_profile(client.client_id)
            profile["full_name"] = client.full_name
            profiles.append(profile)
        profiles.sort(
            key=lambda item: (
                risk_order[str(item["highest_risk"])],
                int(item["blocked_operations_count"]),
                int(item["total_risk_events"]),
                int(item["suspicious_events_count"]),
            ),
            reverse=True,
        )
        return profiles[:limit]

    def _require_matplotlib(self) -> object:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as pyplot
        except ModuleNotFoundError as exc:
            raise InvalidOperationError("matplotlib is required for chart saving") from exc
        return pyplot

    def _event_decimal(self, value: object) -> Decimal:
        try:
            return Decimal(str(value)).quantize(TWOPLACES, rounding=ROUND_HALF_UP)
        except (InvalidOperation, ValueError):
            return Decimal("0.00")

    def _build_bank_balance_timeline(self) -> dict[str, list[float] | list[str]]:
        completed_events = [event for event in self.bank.audit_log.events if event.event_type == "transaction_completed"]
        current_totals = {currency_code: amount for currency_code, amount in self.bank.get_total_balance().items()}
        snapshots: list[dict[str, Decimal]] = [{currency_code: amount for currency_code, amount in current_totals.items()}]
        labels = ["current"]
        for event in reversed(completed_events):
            metadata = event.metadata
            sender_currency = str(metadata.get("sender_currency", metadata.get("currency", "")))
            recipient_currency = str(metadata.get("recipient_currency", metadata.get("currency", "")))
            sender_total_charge = self._event_decimal(metadata.get("sender_total_charge", metadata.get("amount", "0.00")))
            recipient_amount = self._event_decimal(metadata.get("recipient_amount", metadata.get("amount", "0.00")))
            if sender_currency in current_totals:
                current_totals[sender_currency] = (current_totals[sender_currency] + sender_total_charge).quantize(
                    TWOPLACES,
                    rounding=ROUND_HALF_UP,
                )
            if recipient_currency in current_totals:
                current_totals[recipient_currency] = (current_totals[recipient_currency] - recipient_amount).quantize(
                    TWOPLACES,
                    rounding=ROUND_HALF_UP,
                )
            labels.append(event.timestamp.isoformat())
            snapshots.append({currency_code: amount for currency_code, amount in current_totals.items()})
        labels.reverse()
        snapshots.reverse()
        series = {
            currency_code: [float(snapshot[currency_code]) for snapshot in snapshots]
            for currency_code in current_totals
        }
        return {"labels": labels, "series": series}

    def _build_client_balance_timeline(self, client_id: str) -> dict[str, list[float] | list[str]]:
        client = self.bank._get_client(client_id)
        current_totals: dict[str, Decimal] = self.bank._empty_balances_by_currency()
        for account_id in client.account_ids:
            account = self.bank.accounts[account_id]
            current_totals[account.currency.value] += self.bank._account_total_assets(account)
        current_totals = {
            currency_code: amount.quantize(TWOPLACES, rounding=ROUND_HALF_UP)
            for currency_code, amount in current_totals.items()
        }
        related_events = [
            event
            for event in self.bank.audit_log.events
            if event.event_type == "transaction_completed"
            and (
                event.client_id == client.client_id
                or event.metadata.get("recipient_account_id") in client.account_ids
            )
        ]
        snapshots: list[dict[str, Decimal]] = [{currency_code: amount for currency_code, amount in current_totals.items()}]
        labels = ["current"]
        for event in reversed(related_events):
            metadata = event.metadata
            sender_currency = str(metadata.get("sender_currency", metadata.get("currency", "")))
            recipient_currency = str(metadata.get("recipient_currency", metadata.get("currency", "")))
            sender_total_charge = self._event_decimal(metadata.get("sender_total_charge", metadata.get("amount", "0.00")))
            recipient_amount = self._event_decimal(metadata.get("recipient_amount", metadata.get("amount", "0.00")))
            sender_account_id = metadata.get("sender_account_id")
            recipient_account_id = metadata.get("recipient_account_id")
            if sender_account_id in client.account_ids and sender_currency in current_totals:
                current_totals[sender_currency] = (current_totals[sender_currency] + sender_total_charge).quantize(
                    TWOPLACES,
                    rounding=ROUND_HALF_UP,
                )
            if recipient_account_id in client.account_ids and recipient_currency in current_totals:
                current_totals[recipient_currency] = (current_totals[recipient_currency] - recipient_amount).quantize(
                    TWOPLACES,
                    rounding=ROUND_HALF_UP,
                )
            labels.append(event.timestamp.isoformat())
            snapshots.append({currency_code: amount for currency_code, amount in current_totals.items()})
        labels.reverse()
        snapshots.reverse()
        series = {
            currency_code: [float(snapshot[currency_code]) for snapshot in snapshots]
            for currency_code in current_totals
            if any(snapshot[currency_code] != Decimal("0.00") for snapshot in snapshots)
        }
        return {"labels": labels, "series": series}

    def build_client_report(self, client_id: str) -> dict[str, object]:
        client = self.bank._get_client(client_id)
        accounts = [account.get_account_info() for account in self.bank.search_accounts(client_id=client.client_id)]
        audit_events = [
            event.to_dict()
            for event in self.bank.audit_log.events
            if event.client_id == client.client_id
            or event.metadata.get("sender_client_id") == client.client_id
            or event.metadata.get("recipient_client_id") == client.client_id
        ]
        transactions = [
            summary
            for summary in self._build_transaction_summaries()
            if summary["sender_client_id"] == client.client_id or summary["recipient_client_id"] == client.client_id
        ]
        report = {
            "report_type": ReportType.CLIENT.value,
            "generated_at": _utc_now(),
            "bank_name": self.bank.name,
            "client": client.to_safe_dict(),
            "accounts": accounts,
            "total_assets_by_currency": self._build_client_assets_by_currency(client),
            "risk_profile": self.bank.get_client_risk_profile(client.client_id),
            "suspicious_activity": list(client.suspicious_activity),
            "transactions": transactions,
            "audit_events": audit_events,
        }
        return self._serialize_for_export(report)

    def build_bank_report(self) -> dict[str, object]:
        report = {
            "report_type": ReportType.BANK.value,
            "generated_at": _utc_now(),
            "bank_name": self.bank.name,
            "clients_count": len(self.bank.clients),
            "accounts_count": len(self.bank.accounts),
            "total_balance": self.bank.get_total_balance(),
            "client_status_distribution": self._build_client_status_distribution(),
            "account_distribution_by_currency": self._build_account_distribution_by_currency(),
            "account_distribution_by_type": self._build_account_distribution_by_type(),
            "client_rankings": self.bank.get_clients_ranking(),
            "transaction_statistics": self._build_transaction_statistics(),
            "audit_error_statistics": self.bank.get_audit_error_statistics(),
            "audit_events_count": len(self.bank.audit_log.events),
        }
        return self._serialize_for_export(report)

    def build_risk_report(self) -> dict[str, object]:
        blocked_operations = [
            event.to_dict() for event in self.bank.audit_log.filter_events(event_type="transaction_blocked")
        ]
        report = {
            "report_type": ReportType.RISK.value,
            "generated_at": _utc_now(),
            "bank_name": self.bank.name,
            "suspicious_operations": self.bank.get_audit_report_suspicious_operations(RiskLevel.MEDIUM),
            "blocked_operations": blocked_operations,
            "blocked_operations_count": len(blocked_operations),
            "risk_level_distribution": self._build_risk_level_distribution(),
            "top_risky_clients": self._build_top_risky_clients(),
            "audit_error_statistics": self.bank.get_audit_error_statistics(),
        }
        return self._serialize_for_export(report)

    def render_text(self, report: dict[str, object]) -> str:
        if not isinstance(report, dict):
            raise InvalidOperationError("report must be a dictionary")
        return json.dumps(self._serialize_for_export(report), ensure_ascii=False, indent=2)

    def export_to_json(self, report: dict[str, object], file_path: str) -> str:
        if not isinstance(report, dict):
            raise InvalidOperationError("report must be a dictionary")
        path = self._validate_path(file_path, "file_path")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(self._serialize_for_export(report), file, ensure_ascii=False, indent=2)
        return str(path)

    def export_to_csv(self, report: dict[str, object], file_path: str) -> str:
        if not isinstance(report, dict):
            raise InvalidOperationError("report must be a dictionary")
        path = self._validate_path(file_path, "file_path")
        path.parent.mkdir(parents=True, exist_ok=True)
        flattened_rows = self._flatten_report_data(report)
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["field", "value"])
            writer.writerows(flattened_rows)
        return str(path)

    def _save_pie_chart(self, pyplot: object, output_path: Path, title: str, data: dict[str, int]) -> str:
        labels = [label for label, value in data.items() if value > 0]
        sizes = [value for value in data.values() if value > 0]
        pyplot.figure(figsize=(8, 6))
        pyplot.pie(sizes, labels=labels, autopct="%1.1f%%")
        pyplot.title(title)
        pyplot.tight_layout()
        pyplot.savefig(output_path)
        pyplot.close()
        return str(output_path)

    def _save_bar_chart(self, pyplot: object, output_path: Path, title: str, data: dict[str, int]) -> str:
        labels = list(data.keys())
        values = list(data.values())
        pyplot.figure(figsize=(10, 6))
        pyplot.bar(labels, values)
        pyplot.title(title)
        pyplot.xticks(rotation=20)
        pyplot.tight_layout()
        pyplot.savefig(output_path)
        pyplot.close()
        return str(output_path)

    def _save_line_chart(
        self,
        pyplot: object,
        output_path: Path,
        title: str,
        timeline: dict[str, list[float] | list[str]],
    ) -> str:
        labels = list(timeline["labels"])
        pyplot.figure(figsize=(10, 6))
        for series_name, values in dict(timeline["series"]).items():
            pyplot.plot(range(len(values)), values, label=series_name)
        pyplot.title(title)
        pyplot.xlabel("Timeline step")
        pyplot.ylabel("Balance")
        if len(labels) > 1:
            tick_positions = list(range(len(labels)))
            pyplot.xticks(tick_positions, [str(index) for index in tick_positions], rotation=20)
        pyplot.legend()
        pyplot.tight_layout()
        pyplot.savefig(output_path)
        pyplot.close()
        return str(output_path)

    def save_charts(self, output_dir: str, client_id: str | None = None) -> list[str]:
        output_path = self._validate_path(output_dir, "output_dir")
        output_path.mkdir(parents=True, exist_ok=True)
        pyplot = self._require_matplotlib()
        saved_paths = [
            self._save_pie_chart(
                pyplot,
                output_path / "bank_client_status_pie.png",
                "Client Status Distribution",
                self._build_client_status_distribution(),
            ),
            self._save_bar_chart(
                pyplot,
                output_path / "bank_transaction_status_bar.png",
                "Transaction Status Distribution",
                dict(self._build_transaction_statistics()["by_status"]),
            ),
            self._save_bar_chart(
                pyplot,
                output_path / "risk_level_bar.png",
                "Risk Level Distribution",
                self._build_risk_level_distribution(),
            ),
            self._save_line_chart(
                pyplot,
                output_path / "bank_balance_movement.png",
                "Bank Balance Movement",
                self._build_bank_balance_timeline(),
            ),
        ]
        if client_id is not None:
            client = self.bank._get_client(client_id)
            saved_paths.append(
                self._save_line_chart(
                    pyplot,
                    output_path / f"client_{client.client_id}_balance_movement.png",
                    f"Client Balance Movement: {client.full_name}",
                    self._build_client_balance_timeline(client.client_id),
                )
            )
        return saved_paths


class TransactionType(Enum):
    TRANSFER_INTERNAL = "transfer_internal"
    TRANSFER_EXTERNAL = "transfer_external"
    EXCHANGE = "exchange"


class TransactionStatus(Enum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    PROCESSING = "processing"
    RETRYING = "retrying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Transaction:
    transaction_type: TransactionType | str
    amount: Decimal | int | float | str
    currency: Currency | str
    sender_account_id: str
    recipient_account_id: str
    priority: int = 0
    fee: Decimal | int | float | str = Decimal("0.00")
    status: TransactionStatus | str = TransactionStatus.PENDING
    failure_reason: str | None = None
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)
    scheduled_at: datetime | None = None
    processed_at: datetime | None = None
    cancelled_at: datetime | None = None
    failed_at: datetime | None = None
    retry_count: int = 0
    error_log: list[str] = field(default_factory=list)
    transaction_id: str = field(default_factory=lambda: uuid4().hex[:12])

    def __post_init__(self) -> None:
        self.transaction_type = self._coerce_transaction_type(self.transaction_type)
        self.amount = self._normalize_amount(self.amount)
        self.currency = self._coerce_currency(self.currency)
        self.sender_account_id = self._validate_identifier(self.sender_account_id, "sender_account_id")
        self.recipient_account_id = self._validate_identifier(self.recipient_account_id, "recipient_account_id")
        if self.sender_account_id == self.recipient_account_id:
            raise InvalidOperationError("sender and recipient accounts must be different")
        self.priority = self._coerce_priority(self.priority)
        self.fee = self._normalize_amount(self.fee, allow_zero=True)
        self.status = self._coerce_status(self.status)
        self.transaction_id = self._validate_identifier(self.transaction_id, "transaction_id")
        self.created_at = _normalize_utc_datetime(self.created_at, "created_at")
        self.updated_at = _normalize_utc_datetime(self.updated_at, "updated_at")
        if self.scheduled_at is not None:
            self.scheduled_at = _normalize_utc_datetime(self.scheduled_at, "scheduled_at")
        if self.processed_at is not None:
            self.processed_at = _normalize_utc_datetime(self.processed_at, "processed_at")
        if self.cancelled_at is not None:
            self.cancelled_at = _normalize_utc_datetime(self.cancelled_at, "cancelled_at")
        if self.failed_at is not None:
            self.failed_at = _normalize_utc_datetime(self.failed_at, "failed_at")
        if not isinstance(self.retry_count, int) or self.retry_count < 0:
            raise InvalidOperationError("retry_count must be a non-negative integer")
        if not isinstance(self.error_log, list):
            raise InvalidOperationError("error_log must be a list")
        if self.scheduled_at is not None and self.scheduled_at > self.created_at and self.status == TransactionStatus.PENDING:
            self.status = TransactionStatus.SCHEDULED

    @staticmethod
    def _validate_identifier(value: str, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise InvalidOperationError(f"{field_name} must be a non-empty string")
        return value.strip()

    @staticmethod
    def _coerce_priority(priority: int) -> int:
        if not isinstance(priority, int) or isinstance(priority, bool):
            raise InvalidOperationError("priority must be an integer")
        return priority

    @staticmethod
    def _coerce_transaction_type(transaction_type: TransactionType | str) -> TransactionType:
        if isinstance(transaction_type, TransactionType):
            return transaction_type
        if isinstance(transaction_type, str):
            normalized_transaction_type = transaction_type.strip().lower()
            for candidate in TransactionType:
                if candidate.value == normalized_transaction_type:
                    return candidate
        raise InvalidOperationError("unsupported transaction type")

    @staticmethod
    def _coerce_status(status: TransactionStatus | str) -> TransactionStatus:
        if isinstance(status, TransactionStatus):
            return status
        if isinstance(status, str):
            normalized_status = status.strip().lower()
            for candidate in TransactionStatus:
                if candidate.value == normalized_status:
                    return candidate
        raise InvalidOperationError("unsupported transaction status")

    @staticmethod
    def _coerce_currency(currency: Currency | str) -> Currency:
        if isinstance(currency, Currency):
            return currency
        if isinstance(currency, str):
            normalized_currency = currency.strip().upper()
            for candidate in Currency:
                if candidate.value == normalized_currency:
                    return candidate
        raise InvalidOperationError("unsupported currency")

    @staticmethod
    def _normalize_amount(
        amount: Decimal | int | float | str,
        allow_zero: bool = False,
    ) -> Decimal:
        try:
            normalized_amount = Decimal(str(amount)).quantize(TWOPLACES, rounding=ROUND_HALF_UP)
        except (InvalidOperation, ValueError):
            raise InvalidOperationError("amount must be a valid number") from None
        if allow_zero:
            if normalized_amount < 0:
                raise InvalidOperationError("amount cannot be negative")
            return normalized_amount
        if normalized_amount <= 0:
            raise InvalidOperationError("amount must be greater than zero")
        return normalized_amount

    def add_error(self, message: str, occurred_at: datetime | None = None) -> None:
        normalized_message = self._validate_identifier(message, "message")
        self.error_log.append(normalized_message)
        self.updated_at = _utc_now() if occurred_at is None else _normalize_utc_datetime(occurred_at, "occurred_at")

    def mark_processing(self, occurred_at: datetime | None = None) -> None:
        self.status = TransactionStatus.PROCESSING
        self.updated_at = _utc_now() if occurred_at is None else _normalize_utc_datetime(occurred_at, "occurred_at")

    def mark_completed(self, occurred_at: datetime | None = None) -> None:
        timestamp = _utc_now() if occurred_at is None else _normalize_utc_datetime(occurred_at, "occurred_at")
        self.status = TransactionStatus.COMPLETED
        self.failure_reason = None
        self.processed_at = timestamp
        self.updated_at = timestamp

    def mark_failed(self, reason: str, occurred_at: datetime | None = None) -> None:
        timestamp = _utc_now() if occurred_at is None else _normalize_utc_datetime(occurred_at, "occurred_at")
        self.failure_reason = self._validate_identifier(reason, "reason")
        self.status = TransactionStatus.FAILED
        self.failed_at = timestamp
        self.updated_at = timestamp
        self.add_error(self.failure_reason, timestamp)

    def mark_cancelled(self, reason: str, occurred_at: datetime | None = None) -> None:
        timestamp = _utc_now() if occurred_at is None else _normalize_utc_datetime(occurred_at, "occurred_at")
        self.failure_reason = self._validate_identifier(reason, "reason")
        self.status = TransactionStatus.CANCELLED
        self.cancelled_at = timestamp
        self.updated_at = timestamp

    def mark_retrying(self, reason: str, occurred_at: datetime | None = None) -> None:
        timestamp = _utc_now() if occurred_at is None else _normalize_utc_datetime(occurred_at, "occurred_at")
        self.failure_reason = self._validate_identifier(reason, "reason")
        self.status = TransactionStatus.RETRYING
        self.retry_count += 1
        self.updated_at = timestamp
        self.add_error(self.failure_reason, timestamp)


class TransactionQueue:
    def __init__(self) -> None:
        self._transactions: dict[str, Transaction] = {}
        self._sequence_by_id: dict[str, int] = {}
        self._sequence = 0

    def add_transaction(self, transaction: Transaction) -> Transaction:
        if not isinstance(transaction, Transaction):
            raise InvalidOperationError("transaction must be a Transaction instance")
        if transaction.transaction_id in self._transactions:
            raise InvalidOperationError("transaction_id must be unique")
        now = _utc_now()
        if transaction.scheduled_at is not None and transaction.scheduled_at > now:
            transaction.status = TransactionStatus.SCHEDULED
        elif transaction.status == TransactionStatus.SCHEDULED:
            transaction.status = TransactionStatus.PENDING
        transaction.updated_at = now
        self._transactions[transaction.transaction_id] = transaction
        self._sequence_by_id[transaction.transaction_id] = self._sequence
        self._sequence += 1
        return transaction

    def get_transaction(self, transaction_id: str) -> Transaction:
        normalized_transaction_id = Transaction._validate_identifier(transaction_id, "transaction_id")
        if normalized_transaction_id not in self._transactions:
            raise InvalidOperationError("transaction not found")
        return self._transactions[normalized_transaction_id]

    def list_transactions(self) -> list[Transaction]:
        return sorted(
            self._transactions.values(),
            key=lambda transaction: self._sequence_by_id[transaction.transaction_id],
        )

    def cancel_transaction(self, transaction_id: str, reason: str = "cancelled by user") -> Transaction:
        transaction = self.get_transaction(transaction_id)
        if transaction.status in {
            TransactionStatus.PROCESSING,
            TransactionStatus.COMPLETED,
            TransactionStatus.FAILED,
            TransactionStatus.CANCELLED,
        }:
            raise InvalidOperationError("transaction cannot be cancelled")
        transaction.mark_cancelled(reason)
        return transaction

    def get_ready_transactions(self, now: datetime | None = None) -> list[Transaction]:
        current_time = _utc_now() if now is None else _normalize_utc_datetime(now, "now")
        ready_transactions = [
            transaction
            for transaction in self._transactions.values()
            if transaction.status in {TransactionStatus.PENDING, TransactionStatus.RETRYING}
            or (
                transaction.status == TransactionStatus.SCHEDULED
                and transaction.scheduled_at is not None
                and transaction.scheduled_at <= current_time
            )
        ]
        return sorted(
            ready_transactions,
            key=lambda transaction: (
                -transaction.priority,
                self._sequence_by_id[transaction.transaction_id],
            ),
        )


class TransactionProcessor:
    def __init__(
        self,
        bank: Bank,
        exchange_rates: dict[tuple[str, str], Decimal | int | float | str] | None = None,
        external_transfer_fee: Decimal | int | float | str = Decimal("10.00"),
        max_retries: int = 3,
    ) -> None:
        if not isinstance(bank, Bank):
            raise InvalidOperationError("bank must be a Bank instance")
        if not isinstance(max_retries, int) or max_retries < 0:
            raise InvalidOperationError("max_retries must be a non-negative integer")
        self.bank = bank
        self.max_retries = max_retries
        self.external_transfer_fee = Transaction._normalize_amount(external_transfer_fee, allow_zero=True)
        self.exchange_rates = self._normalize_exchange_rates(exchange_rates or {})

    def _normalize_exchange_rates(
        self,
        exchange_rates: dict[tuple[str, str], Decimal | int | float | str],
    ) -> dict[tuple[str, str], Decimal]:
        normalized_rates: dict[tuple[str, str], Decimal] = {}
        for currency_pair, rate in exchange_rates.items():
            if not isinstance(currency_pair, tuple) or len(currency_pair) != 2:
                raise InvalidOperationError("exchange rate key must be a pair of currency codes")
            source_currency = Transaction._coerce_currency(currency_pair[0]).value
            target_currency = Transaction._coerce_currency(currency_pair[1]).value
            normalized_rate = Transaction._normalize_amount(rate)
            normalized_rates[(source_currency, target_currency)] = normalized_rate
        return normalized_rates

    def _get_exchange_rate(self, source_currency: Currency, target_currency: Currency) -> Decimal:
        if source_currency == target_currency:
            return Decimal("1.00")
        direct_pair = (source_currency.value, target_currency.value)
        if direct_pair in self.exchange_rates:
            return self.exchange_rates[direct_pair]
        reverse_pair = (target_currency.value, source_currency.value)
        if reverse_pair in self.exchange_rates:
            reverse_rate = self.exchange_rates[reverse_pair]
            return (Decimal("1.00") / reverse_rate).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        raise InvalidOperationError("exchange rate not found")

    def _convert_amount(self, amount: Decimal, source_currency: Currency, target_currency: Currency) -> Decimal:
        exchange_rate = self._get_exchange_rate(source_currency, target_currency)
        return (amount * exchange_rate).quantize(TWOPLACES, rounding=ROUND_HALF_UP)

    def _ensure_account_can_transact(self, account_id: str, role: str) -> None:
        account = self.bank._get_account(account_id)
        if account.status == AccountStatus.FROZEN:
            raise AccountFrozenError(f"{role} account is frozen")
        if account.status == AccountStatus.CLOSED:
            raise AccountClosedError(f"{role} account is closed")

    def _ensure_sender_client_can_transact(self, account_id: str) -> None:
        client = self.bank._get_account_owner_client(account_id)
        if client.is_locked or client.status == ClientStatus.BLOCKED:
            raise InvalidOperationError("blocked client cannot initiate transaction")

    def _validate_transaction_type(self, transaction: Transaction) -> tuple[str, str, bool]:
        sender_client_id = self.bank.account_to_client[transaction.sender_account_id]
        recipient_client_id = self.bank.account_to_client[transaction.recipient_account_id]
        same_client = sender_client_id == recipient_client_id
        sender_account = self.bank._get_account(transaction.sender_account_id)
        recipient_account = self.bank._get_account(transaction.recipient_account_id)
        if transaction.transaction_type == TransactionType.TRANSFER_INTERNAL:
            if not same_client:
                raise InvalidOperationError("internal transfer requires accounts of the same client")
            if sender_account.currency != recipient_account.currency:
                raise InvalidOperationError("internal transfer requires matching account currencies")
        if transaction.transaction_type == TransactionType.TRANSFER_EXTERNAL and same_client:
            raise InvalidOperationError("external transfer requires accounts of different clients")
        if transaction.transaction_type == TransactionType.EXCHANGE:
            if not same_client:
                raise InvalidOperationError("exchange requires accounts of the same client")
            if sender_account.currency == recipient_account.currency:
                raise InvalidOperationError("exchange requires different account currencies")
        return sender_client_id, recipient_client_id, same_client

    def _calculate_fee(self, same_client: bool) -> Decimal:
        if same_client:
            return Decimal("0.00")
        return self.external_transfer_fee.quantize(TWOPLACES, rounding=ROUND_HALF_UP)

    @staticmethod
    def _calculate_account_fee(sender_account: AbstractAccount) -> Decimal:
        if isinstance(sender_account, PremiumAccount):
            return sender_account.fixed_commission.quantize(TWOPLACES, rounding=ROUND_HALF_UP)
        return Decimal("0.00")

    def _ensure_transaction_time_allowed(self, transaction: Transaction, current_time: datetime) -> None:
        sender_client = self.bank._get_account_owner_client(transaction.sender_account_id)
        action_name = f"transaction_{transaction.transaction_type.value}"
        self.bank._ensure_operation_time_allowed(sender_client, action_name, current_time)

    @staticmethod
    def _risk_severity_for_level(risk_level: RiskLevel) -> AuditSeverity:
        if risk_level == RiskLevel.HIGH:
            return AuditSeverity.HIGH
        if risk_level == RiskLevel.MEDIUM:
            return AuditSeverity.MEDIUM
        return AuditSeverity.LOW

    def _build_audit_metadata(
        self,
        transaction: Transaction,
        risk_assessment: RiskAssessment | None = None,
        execution_details: dict[str, object] | None = None,
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "transaction_type": transaction.transaction_type.value,
            "amount": f"{transaction.amount:.2f}",
            "currency": transaction.currency.value,
            "sender_account_id": transaction.sender_account_id,
            "recipient_account_id": transaction.recipient_account_id,
            "fee": f"{transaction.fee:.2f}",
        }
        sender_client_id = self.bank.account_to_client.get(transaction.sender_account_id)
        recipient_client_id = self.bank.account_to_client.get(transaction.recipient_account_id)
        if sender_client_id is not None:
            metadata["sender_client_id"] = sender_client_id
        if recipient_client_id is not None:
            metadata["recipient_client_id"] = recipient_client_id
        if execution_details is not None:
            metadata.update(execution_details)
        if risk_assessment is not None:
            metadata["risk_reasons"] = list(risk_assessment.reasons)
            metadata["risk_score"] = risk_assessment.score
        return metadata

    def _log_risk_assessment(
        self,
        transaction: Transaction,
        client_id: str,
        risk_assessment: RiskAssessment,
        current_time: datetime,
    ) -> None:
        if not risk_assessment.reasons:
            return
        self.bank.log_audit_event(
            severity=self._risk_severity_for_level(risk_assessment.risk_level),
            event_type="risk_assessment",
            message=", ".join(risk_assessment.reasons),
            client_id=client_id,
            account_id=transaction.sender_account_id,
            transaction_id=transaction.transaction_id,
            risk_level=risk_assessment.risk_level,
            metadata=self._build_audit_metadata(transaction, risk_assessment),
            timestamp=current_time,
        )

    def _execute_transaction(self, transaction: Transaction) -> dict[str, object]:
        sender_account = self.bank._get_account(transaction.sender_account_id)
        recipient_account = self.bank._get_account(transaction.recipient_account_id)
        self._ensure_account_can_transact(sender_account.account_id, "sender")
        self._ensure_account_can_transact(recipient_account.account_id, "recipient")
        self._ensure_sender_client_can_transact(sender_account.account_id)
        _, _, same_client = self._validate_transaction_type(transaction)
        if transaction.currency != sender_account.currency:
            raise InvalidOperationError("transaction currency must match sender account currency")
        processor_fee = self._calculate_fee(same_client)
        account_fee = self._calculate_account_fee(sender_account)
        transaction.fee = (processor_fee + account_fee).quantize(TWOPLACES, rounding=ROUND_HALF_UP)
        recipient_amount = self._convert_amount(transaction.amount, sender_account.currency, recipient_account.currency)
        sender_charge_before_account_fee = (transaction.amount + processor_fee).quantize(TWOPLACES, rounding=ROUND_HALF_UP)
        total_sender_charge = (sender_charge_before_account_fee + account_fee).quantize(TWOPLACES, rounding=ROUND_HALF_UP)
        sender_account.withdraw(sender_charge_before_account_fee)
        recipient_account.deposit(recipient_amount)
        return {
            "sender_currency": sender_account.currency.value,
            "recipient_currency": recipient_account.currency.value,
            "processor_fee": f"{processor_fee:.2f}",
            "account_fee": f"{account_fee:.2f}",
            "total_fee": f"{transaction.fee:.2f}",
            "sender_total_charge": f"{total_sender_charge:.2f}",
            "recipient_amount": f"{recipient_amount:.2f}",
        }

    def process_transaction(self, transaction: Transaction, now: datetime | None = None) -> Transaction:
        if not isinstance(transaction, Transaction):
            raise InvalidOperationError("transaction must be a Transaction instance")
        current_time = _utc_now() if now is None else _normalize_utc_datetime(now, "now")
        if transaction.status in {TransactionStatus.COMPLETED, TransactionStatus.CANCELLED, TransactionStatus.FAILED}:
            return transaction
        if transaction.scheduled_at is not None and transaction.scheduled_at > current_time:
            transaction.status = TransactionStatus.SCHEDULED
            transaction.updated_at = current_time
            return transaction
        sender_client: Client | None = None
        risk_assessment: RiskAssessment | None = None
        execution_details: dict[str, object] | None = None
        try:
            sender_client = self.bank._get_account_owner_client(transaction.sender_account_id)
            risk_assessment = self.bank.risk_analyzer.analyze_transaction(transaction, self.bank, current_time)
            self._log_risk_assessment(transaction, sender_client.client_id, risk_assessment, current_time)
            if risk_assessment.should_block:
                reason = "operation blocked by risk analyzer"
                self.bank._mark_suspicious(sender_client, f"risk blocked operation: {transaction.transaction_type.value}")
                self.bank.log_audit_event(
                    severity=self._risk_severity_for_level(risk_assessment.risk_level),
                    event_type="transaction_blocked",
                    message=reason,
                    client_id=sender_client.client_id,
                    account_id=transaction.sender_account_id,
                    transaction_id=transaction.transaction_id,
                    risk_level=risk_assessment.risk_level,
                    metadata=self._build_audit_metadata(transaction, risk_assessment),
                    timestamp=current_time,
                )
                raise InvalidOperationError(reason)
            self._ensure_transaction_time_allowed(transaction, current_time)
            transaction.mark_processing(current_time)
            execution_details = self._execute_transaction(transaction)
        except (InvalidOperationError, AccountFrozenError, AccountClosedError, InsufficientFundsError) as exc:
            if risk_assessment is None or not risk_assessment.should_block:
                self.bank.log_audit_event(
                    severity=AuditSeverity.MEDIUM,
                    event_type="transaction_failed",
                    message=str(exc),
                    client_id=sender_client.client_id if sender_client is not None else None,
                    account_id=transaction.sender_account_id,
                    transaction_id=transaction.transaction_id,
                    risk_level=risk_assessment.risk_level if risk_assessment is not None and risk_assessment.reasons else None,
                    metadata=self._build_audit_metadata(transaction, risk_assessment, execution_details),
                    timestamp=current_time,
                )
            transaction.mark_failed(str(exc), current_time)
            return transaction
        except Exception as exc:
            self.bank.log_audit_event(
                severity=AuditSeverity.HIGH,
                event_type="operation_error",
                message=str(exc),
                client_id=sender_client.client_id if sender_client is not None else None,
                account_id=transaction.sender_account_id,
                transaction_id=transaction.transaction_id,
                risk_level=risk_assessment.risk_level if risk_assessment is not None and risk_assessment.reasons else None,
                metadata=self._build_audit_metadata(transaction, risk_assessment, execution_details),
                timestamp=current_time,
            )
            if transaction.retry_count < self.max_retries:
                transaction.mark_retrying(str(exc), current_time)
                return transaction
            transaction.mark_failed(str(exc), current_time)
            return transaction
        self.bank.log_audit_event(
            severity=AuditSeverity.LOW,
            event_type="transaction_completed",
            message="transaction completed successfully",
            client_id=sender_client.client_id,
            account_id=transaction.sender_account_id,
            transaction_id=transaction.transaction_id,
            risk_level=risk_assessment.risk_level if risk_assessment.reasons else None,
            metadata=self._build_audit_metadata(transaction, risk_assessment, execution_details),
            timestamp=current_time,
        )
        transaction.mark_completed(current_time)
        return transaction

    def process_queue(
        self,
        transaction_queue: TransactionQueue,
        now: datetime | None = None,
        limit: int | None = None,
    ) -> list[Transaction]:
        if not isinstance(transaction_queue, TransactionQueue):
            raise InvalidOperationError("transaction_queue must be a TransactionQueue instance")
        ready_transactions = transaction_queue.get_ready_transactions(now)
        if limit is not None:
            if not isinstance(limit, int) or limit <= 0:
                raise InvalidOperationError("limit must be a positive integer")
            ready_transactions = ready_transactions[:limit]
        processed_transactions: list[Transaction] = []
        for transaction in ready_transactions:
            processed_transactions.append(self.process_transaction(transaction, now))
        return processed_transactions

    def process_until_idle(
        self,
        transaction_queue: TransactionQueue,
        now: datetime | None = None,
        max_cycles: int | None = None,
    ) -> list[Transaction]:
        if max_cycles is not None and (not isinstance(max_cycles, int) or max_cycles <= 0):
            raise InvalidOperationError("max_cycles must be a positive integer")
        processed_transactions: list[Transaction] = []
        processed_transaction_ids: set[str] = set()
        current_cycle = 0
        while True:
            current_cycle += 1
            if max_cycles is not None and current_cycle > max_cycles:
                break
            batch = [
                transaction
                for transaction in transaction_queue.get_ready_transactions(now)
                if transaction.transaction_id not in processed_transaction_ids
            ]
            if not batch:
                break
            for transaction in batch:
                processed_transactions.append(self.process_transaction(transaction, now))
                processed_transaction_ids.add(transaction.transaction_id)
            if all(transaction.status != TransactionStatus.RETRYING for transaction in processed_transactions):
                break
        return processed_transactions


class HTMLParser:
    def __init__(self, same_domain_only: bool = False) -> None:
        self.same_domain_only = same_domain_only

    @staticmethod
    def empty_result(url: str) -> dict[str, object]:
        return {
            "url": url,
            "title": "",
            "text": "",
            "links": [],
            "metadata": {},
            "images": [],
            "headings": [],
            "tables": [],
            "lists": [],
        }

    @staticmethod
    def _normalize_text(value: str) -> str:
        return " ".join(value.split())

    @staticmethod
    def _is_valid_absolute_url(url: str) -> bool:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    def _normalize_url(self, candidate: str, base_url: str) -> str:
        if not isinstance(candidate, str) or not candidate.strip():
            return ""
        normalized_candidate = candidate.strip()
        lowered_candidate = normalized_candidate.lower()
        if lowered_candidate.startswith(("javascript:", "mailto:", "tel:")):
            return ""
        if normalized_candidate.startswith("#"):
            return ""
        normalized_url, _fragment = urldefrag(urljoin(base_url, normalized_candidate))
        if not self._is_valid_absolute_url(normalized_url):
            return ""
        if self.same_domain_only and urlparse(normalized_url).netloc != urlparse(base_url).netloc:
            return ""
        return normalized_url

    def extract_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        unique_links: list[str] = []
        seen_links: set[str] = set()
        for tag in soup.find_all("a", href=True):
            normalized_url = self._normalize_url(str(tag.get("href", "")), base_url)
            if not normalized_url or normalized_url in seen_links:
                continue
            seen_links.add(normalized_url)
            unique_links.append(normalized_url)
        return unique_links

    def extract_text(self, soup: BeautifulSoup, selector: str = None) -> str:
        try:
            target = soup.select_one(selector) if selector else (soup.body or soup)
        except Exception as error:
            logger.warning("Text extraction failed for selector %s: %s", selector, error)
            return ""
        if target is None:
            return ""
        return self._normalize_text(target.get_text(" ", strip=True))

    def extract_metadata(self, soup: BeautifulSoup) -> dict[str, str]:
        title = ""
        if soup.title is not None:
            title = self._normalize_text(next(soup.title.stripped_strings, ""))
        description_tag = soup.find("meta", attrs={"name": lambda value: isinstance(value, str) and value.lower() == "description"})
        keywords_tag = soup.find("meta", attrs={"name": lambda value: isinstance(value, str) and value.lower() == "keywords"})
        return {
            "title": title,
            "description": self._normalize_text(str(description_tag.get("content", ""))) if description_tag else "",
            "keywords": self._normalize_text(str(keywords_tag.get("content", ""))) if keywords_tag else "",
        }

    def extract_images(self, soup: BeautifulSoup, base_url: str) -> list[dict[str, str]]:
        images: list[dict[str, str]] = []
        for image in soup.find_all("img"):
            normalized_src = self._normalize_url(str(image.get("src", "")), base_url)
            if not normalized_src:
                continue
            images.append(
                {
                    "src": normalized_src,
                    "alt": self._normalize_text(str(image.get("alt", ""))),
                }
            )
        return images

    def extract_headings(self, soup: BeautifulSoup) -> list[dict[str, str]]:
        headings: list[dict[str, str]] = []
        for tag in soup.find_all(["h1", "h2", "h3"]):
            text = self._normalize_text(tag.get_text(" ", strip=True))
            if not text:
                continue
            headings.append({"tag": tag.name, "text": text})
        return headings

    def extract_tables(self, soup: BeautifulSoup) -> list[list[list[str]]]:
        tables: list[list[list[str]]] = []
        for table in soup.find_all("table"):
            rows: list[list[str]] = []
            for row in table.find_all("tr"):
                cells = [self._normalize_text(cell.get_text(" ", strip=True)) for cell in row.find_all(["th", "td"])]
                cells = [cell for cell in cells if cell]
                if cells:
                    rows.append(cells)
            if rows:
                tables.append(rows)
        return tables

    def extract_lists(self, soup: BeautifulSoup) -> list[dict[str, object]]:
        lists: list[dict[str, object]] = []
        for list_tag in soup.find_all(["ul", "ol"]):
            items = [self._normalize_text(item.get_text(" ", strip=True)) for item in list_tag.find_all("li")]
            items = [item for item in items if item]
            if items:
                lists.append({"type": list_tag.name, "items": items})
        return lists

    async def parse_html(self, html: str, url: str) -> dict[str, object]:
        result = self.empty_result(url)
        if not isinstance(html, str) or not html:
            logger.warning("Empty or invalid HTML received for %s", url)
            return result
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as error:
            logger.warning("Failed to parse HTML for %s: %s", url, error)
            return result
        try:
            metadata = self.extract_metadata(soup)
            result["metadata"] = metadata
            result["title"] = metadata.get("title", "")
        except Exception as error:
            logger.warning("Metadata extraction failed for %s: %s", url, error)
        try:
            result["text"] = self.extract_text(soup)
        except Exception as error:
            logger.warning("Text extraction failed for %s: %s", url, error)
        try:
            result["links"] = self.extract_links(soup, url)
        except Exception as error:
            logger.warning("Link extraction failed for %s: %s", url, error)
        try:
            result["images"] = self.extract_images(soup, url)
        except Exception as error:
            logger.warning("Image extraction failed for %s: %s", url, error)
        try:
            result["headings"] = self.extract_headings(soup)
        except Exception as error:
            logger.warning("Heading extraction failed for %s: %s", url, error)
        try:
            result["tables"] = self.extract_tables(soup)
        except Exception as error:
            logger.warning("Table extraction failed for %s: %s", url, error)
        try:
            result["lists"] = self.extract_lists(soup)
        except Exception as error:
            logger.warning("List extraction failed for %s: %s", url, error)
        return result


class CrawlerQueue:
    def __init__(self) -> None:
        self._heap: list[tuple[int, int, str, int]] = []
        self._counter = 0
        self._lock = asyncio.Lock()
        self._queued_urls: set[str] = set()
        self._in_progress_urls: set[str] = set()
        self._processed_urls: set[str] = set()
        self._failed_urls: dict[str, str] = {}
        self._depths: dict[str, int] = {}

    async def add_url(self, url: str, priority: int = 0, depth: int = 0) -> bool:
        if not isinstance(url, str) or not url.strip():
            raise InvalidOperationError("url must be a non-empty string")
        if not isinstance(priority, int):
            raise InvalidOperationError("priority must be an integer")
        if not isinstance(depth, int) or depth < 0:
            raise InvalidOperationError("depth must be a non-negative integer")
        normalized_url = url.strip()
        async with self._lock:
            if normalized_url in self._queued_urls:
                return False
            if normalized_url in self._in_progress_urls:
                return False
            if normalized_url in self._processed_urls:
                return False
            if normalized_url in self._failed_urls:
                return False
            heapq.heappush(self._heap, (priority, self._counter, normalized_url, depth))
            self._counter += 1
            self._queued_urls.add(normalized_url)
            self._depths[normalized_url] = depth
            return True

    async def get_next(self) -> str | None:
        async with self._lock:
            if not self._heap:
                return None
            _priority, _counter, url, _depth = heapq.heappop(self._heap)
            self._queued_urls.discard(url)
            self._in_progress_urls.add(url)
            return url

    def get_depth(self, url: str) -> int:
        return self._depths.get(url, 0)

    def is_known(self, url: str) -> bool:
        return url in self._queued_urls or url in self._in_progress_urls or url in self._processed_urls or url in self._failed_urls

    def pending_count(self) -> int:
        return len(self._queued_urls)

    def mark_processed(self, url: str) -> None:
        normalized_url = url.strip()
        self._in_progress_urls.discard(normalized_url)
        self._queued_urls.discard(normalized_url)
        self._failed_urls.pop(normalized_url, None)
        self._processed_urls.add(normalized_url)

    def mark_failed(self, url: str, error: str) -> None:
        normalized_url = url.strip()
        self._in_progress_urls.discard(normalized_url)
        self._queued_urls.discard(normalized_url)
        self._failed_urls[normalized_url] = error.strip() if isinstance(error, str) and error.strip() else "unknown error"

    def get_stats(self) -> dict[str, object]:
        return {
            "pending": len(self._queued_urls),
            "in_progress": len(self._in_progress_urls),
            "processed": len(self._processed_urls),
            "failed": len(self._failed_urls),
            "known": len(self._queued_urls | self._in_progress_urls | self._processed_urls | set(self._failed_urls.keys())),
        }


class SemaphoreManager:
    def __init__(self, max_concurrent: int, per_domain_concurrent: int | None = None) -> None:
        if not isinstance(max_concurrent, int) or max_concurrent < 1:
            raise InvalidOperationError("max_concurrent must be a positive integer")
        if per_domain_concurrent is None:
            per_domain_concurrent = max_concurrent
        if not isinstance(per_domain_concurrent, int) or per_domain_concurrent < 1:
            raise InvalidOperationError("per_domain_concurrent must be a positive integer")
        self.max_concurrent = max_concurrent
        self.per_domain_concurrent = per_domain_concurrent
        self._global_semaphore = asyncio.Semaphore(max_concurrent)
        self._domain_semaphores: dict[str, asyncio.Semaphore] = {}
        self._lock = asyncio.Lock()
        self._active_tasks = 0
        self._active_by_domain: dict[str, int] = {}

    @staticmethod
    def _get_domain(url: str) -> str:
        return urlparse(url).netloc

    async def _get_domain_semaphore(self, domain: str) -> asyncio.Semaphore:
        async with self._lock:
            semaphore = self._domain_semaphores.get(domain)
            if semaphore is None:
                semaphore = asyncio.Semaphore(self.per_domain_concurrent)
                self._domain_semaphores[domain] = semaphore
            return semaphore

    @asynccontextmanager
    async def limit(self, url: str):
        domain = self._get_domain(url)
        domain_semaphore = await self._get_domain_semaphore(domain)
        await self._global_semaphore.acquire()
        await domain_semaphore.acquire()
        async with self._lock:
            self._active_tasks += 1
            self._active_by_domain[domain] = self._active_by_domain.get(domain, 0) + 1
        try:
            yield
        finally:
            async with self._lock:
                self._active_tasks = max(0, self._active_tasks - 1)
                current_active = self._active_by_domain.get(domain, 0) - 1
                if current_active <= 0:
                    self._active_by_domain.pop(domain, None)
                else:
                    self._active_by_domain[domain] = current_active
            domain_semaphore.release()
            self._global_semaphore.release()

    def get_stats(self) -> dict[str, object]:
        return {
            "max_concurrent": self.max_concurrent,
            "per_domain_concurrent": self.per_domain_concurrent,
            "active_tasks": self._active_tasks,
            "active_by_domain": dict(self._active_by_domain),
        }


class RateLimiter:
    def __init__(self, requests_per_second: float = 1.0, per_domain: bool = True) -> None:
        if not isinstance(requests_per_second, (int, float)) or requests_per_second <= 0:
            raise InvalidOperationError("requests_per_second must be a positive number")
        if not isinstance(per_domain, bool):
            raise InvalidOperationError("per_domain must be a boolean")
        self.requests_per_second = float(requests_per_second)
        self.per_domain = per_domain
        self._interval = 1.0 / self.requests_per_second
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()
        self._last_request_at: dict[str, float] = {}
        self._started_at = perf_counter()
        self._acquire_count = 0
        self._total_wait_time = 0.0
        self._recent_acquires: deque[float] = deque()
        self._stats_lock = asyncio.Lock()

    async def _get_lock(self, key: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
            return lock

    async def acquire(self, domain: str = None) -> float:
        key = domain if self.per_domain and isinstance(domain, str) and domain else "__global__"
        lock = await self._get_lock(key)
        async with lock:
            now = perf_counter()
            last_request_at = self._last_request_at.get(key)
            wait_time = 0.0
            if last_request_at is not None:
                wait_time = max(0.0, (last_request_at + self._interval) - now)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            acquired_at = perf_counter()
            self._last_request_at[key] = acquired_at
            async with self._stats_lock:
                self._acquire_count += 1
                self._total_wait_time += wait_time
                self._recent_acquires.append(acquired_at)
                window_start = acquired_at - 1.0
                while self._recent_acquires and self._recent_acquires[0] < window_start:
                    self._recent_acquires.popleft()
            return wait_time

    def get_stats(self) -> dict[str, object]:
        now = perf_counter()
        while self._recent_acquires and self._recent_acquires[0] < now - 1.0:
            self._recent_acquires.popleft()
        elapsed = now - self._started_at if self._started_at else 0.0
        return {
            "requests_per_second": self.requests_per_second,
            "per_domain": self.per_domain,
            "interval": self._interval,
            "acquire_count": self._acquire_count,
            "current_req_per_sec": float(len(self._recent_acquires)),
            "average_wait": self._total_wait_time / self._acquire_count if self._acquire_count else 0.0,
            "overall_req_per_sec": self._acquire_count / elapsed if elapsed > 0 else 0.0,
        }


class RobotsParser:
    def __init__(self, timeout: float = 5.0, user_agent: str = "AsyncCrawler/1.0") -> None:
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise InvalidOperationError("timeout must be a positive number")
        if not isinstance(user_agent, str) or not user_agent.strip():
            raise InvalidOperationError("user_agent must be a non-empty string")
        self.timeout = float(timeout)
        self.user_agent = user_agent.strip()
        self._cache: dict[str, dict[str, object]] = {}
        self._cache_lock = asyncio.Lock()
        self._active_base_url = ""

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        parsed = urlparse(base_url)
        if not parsed.scheme or not parsed.netloc:
            return ""
        return f"{parsed.scheme}://{parsed.netloc}"

    @staticmethod
    def _path_from_url(url: str) -> str:
        parsed = urlparse(url)
        path = parsed.path or "/"
        if parsed.query:
            return f"{path}?{parsed.query}"
        return path

    @staticmethod
    def _normalize_user_agent(user_agent: str) -> str:
        return user_agent.strip().lower() if isinstance(user_agent, str) and user_agent.strip() else "*"

    @staticmethod
    def _finalize_robot_group(group: dict[str, object], rules: dict[str, dict[str, object]]) -> None:
        user_agents = list(group.get("user_agents", []))
        if not user_agents:
            return
        for user_agent in user_agents:
            if user_agent not in rules:
                rules[user_agent] = {"allow": [], "disallow": [], "crawl_delay": None}
            rules[user_agent]["allow"].extend(list(group.get("allow", [])))
            rules[user_agent]["disallow"].extend(list(group.get("disallow", [])))
            crawl_delay = group.get("crawl_delay")
            if crawl_delay is not None:
                rules[user_agent]["crawl_delay"] = crawl_delay

    def _parse_robots_text(self, base_url: str, content: str) -> dict[str, object]:
        rules: dict[str, dict[str, object]] = {}
        current_group: dict[str, object] = {"user_agents": [], "allow": [], "disallow": [], "crawl_delay": None}
        seen_directive = False
        for raw_line in content.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                if current_group["user_agents"]:
                    self._finalize_robot_group(current_group, rules)
                    current_group = {"user_agents": [], "allow": [], "disallow": [], "crawl_delay": None}
                    seen_directive = False
                continue
            if ":" not in line:
                continue
            field, value = line.split(":", 1)
            directive = field.strip().lower()
            directive_value = value.strip()
            if directive == "user-agent":
                if seen_directive and current_group["user_agents"]:
                    self._finalize_robot_group(current_group, rules)
                    current_group = {"user_agents": [], "allow": [], "disallow": [], "crawl_delay": None}
                    seen_directive = False
                current_group["user_agents"].append(self._normalize_user_agent(directive_value))
                continue
            if not current_group["user_agents"]:
                continue
            seen_directive = True
            if directive == "allow":
                current_group["allow"].append(directive_value or "/")
            elif directive == "disallow":
                current_group["disallow"].append(directive_value)
            elif directive == "crawl-delay":
                try:
                    current_group["crawl_delay"] = max(0.0, float(directive_value))
                except ValueError:
                    continue
        if current_group["user_agents"]:
            self._finalize_robot_group(current_group, rules)
        return {
            "base_url": base_url,
            "rules": rules,
            "fetched": True,
            "allow_all": False,
        }

    async def fetch_robots(self, base_url: str) -> dict:
        normalized_base_url = self._normalize_base_url(base_url)
        if not normalized_base_url:
            return {"base_url": base_url, "rules": {}, "fetched": False, "allow_all": True}
        self._active_base_url = normalized_base_url
        async with self._cache_lock:
            cached_rules = self._cache.get(normalized_base_url)
            if cached_rules is not None:
                return cached_rules
        robots_url = urljoin(f"{normalized_base_url}/", "robots.txt")
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": self.user_agent}) as session:
                async with session.get(robots_url) as response:
                    if response.status == 404:
                        rules = {"base_url": normalized_base_url, "rules": {}, "fetched": True, "allow_all": True}
                    elif response.status >= 400:
                        rules = {"base_url": normalized_base_url, "rules": {}, "fetched": False, "allow_all": True}
                    else:
                        content = await response.text()
                        rules = self._parse_robots_text(normalized_base_url, content)
        except (aiohttp.ClientError, asyncio.TimeoutError) as error:
            logger.warning("Failed to fetch robots.txt for %s: %s", normalized_base_url, error)
            rules = {"base_url": normalized_base_url, "rules": {}, "fetched": False, "allow_all": True}
        async with self._cache_lock:
            self._cache[normalized_base_url] = rules
        return rules

    def _select_rules(self, domain: str, user_agent: str) -> dict[str, object]:
        if not domain:
            return {"allow": [], "disallow": [], "crawl_delay": None}
        matching_cache = self._cache.get(domain)
        if matching_cache is None:
            normalized_domain = self._normalize_base_url(domain)
            matching_cache = self._cache.get(normalized_domain, {"rules": {}})
        rules = dict(matching_cache.get("rules", {}))
        normalized_user_agent = self._normalize_user_agent(user_agent)
        if normalized_user_agent in rules:
            return dict(rules[normalized_user_agent])
        for rule_user_agent, rule_group in rules.items():
            if rule_user_agent != "*" and normalized_user_agent.startswith(rule_user_agent):
                return dict(rule_group)
        return dict(rules.get("*", {"allow": [], "disallow": [], "crawl_delay": None}))

    def can_fetch(self, url: str, user_agent: str = "*") -> bool:
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else ""
        if not domain:
            return True
        self._active_base_url = domain
        rule_group = self._select_rules(domain, user_agent)
        path = self._path_from_url(url)
        best_allow_length = -1
        best_disallow_length = -1
        for allow_rule in list(rule_group.get("allow", [])):
            if allow_rule and path.startswith(allow_rule):
                best_allow_length = max(best_allow_length, len(allow_rule))
        for disallow_rule in list(rule_group.get("disallow", [])):
            if disallow_rule and path.startswith(disallow_rule):
                best_disallow_length = max(best_disallow_length, len(disallow_rule))
        if best_disallow_length < 0:
            return True
        return best_allow_length >= best_disallow_length

    def _get_crawl_delay_for_domain(self, domain: str, user_agent: str = "*") -> float:
        rule_group = self._select_rules(domain, user_agent)
        crawl_delay = rule_group.get("crawl_delay")
        return float(crawl_delay) if isinstance(crawl_delay, (int, float)) else 0.0

    def get_crawl_delay(self, user_agent: str = "*") -> float:
        active_base_url = self._normalize_base_url(self._active_base_url)
        if not active_base_url:
            return 0.0
        return self._get_crawl_delay_for_domain(active_base_url, user_agent=user_agent)

    def get_stats(self) -> dict[str, object]:
        return {
            "cached_domains": len(self._cache),
            "domains": sorted(self._cache.keys()),
        }


class CrawlerError(Exception):
    def __init__(
        self,
        message: str,
        url: str = "",
        original_error: Exception | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.url = url
        self.original_error = original_error
        self.status_code = status_code


class TransientError(CrawlerError):
    pass


class PermanentError(CrawlerError):
    pass


class NetworkError(CrawlerError):
    pass


class ParseError(CrawlerError):
    pass


class DataStorage(ABC):
    @abstractmethod
    async def save(self, data: dict) -> None:
        raise NotImplementedError()

    async def flush(self) -> None:
        return None

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get_stats(self) -> dict[str, object]:
        raise NotImplementedError()


class JSONStorage(DataStorage):
    def __init__(
        self,
        file_path: str,
        encoding: str = "utf-8",
        pretty: bool = False,
        buffer_size: int = 1,
        retry_strategy: RetryStrategy | None = None,
    ) -> None:
        if not isinstance(file_path, str) or not file_path.strip():
            raise InvalidOperationError("file_path must be a non-empty string")
        if not isinstance(encoding, str) or not encoding.strip():
            raise InvalidOperationError("encoding must be a non-empty string")
        if not isinstance(pretty, bool):
            raise InvalidOperationError("pretty must be a boolean")
        if not isinstance(buffer_size, int) or buffer_size < 1:
            raise InvalidOperationError("buffer_size must be a positive integer")
        if retry_strategy is not None and not isinstance(retry_strategy, RetryStrategy):
            raise InvalidOperationError("retry_strategy must be an instance of RetryStrategy")
        self.file_path = Path(file_path).expanduser()
        self.encoding = encoding.strip()
        self.pretty = pretty
        self.buffer_size = buffer_size
        self.retry_strategy = retry_strategy if retry_strategy is not None else RetryStrategy(max_retries=1, base_delay=0.1)
        self._buffer: list[dict[str, object]] = []
        self._lock = asyncio.Lock()
        self._stats = {"saved_count": 0, "flush_count": 0, "errors": 0, "path": str(self.file_path)}

    def _serialize_line(self, record: dict[str, object]) -> str:
        if self.pretty:
            return json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        return json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"

    async def _write_lines(self, records: list[dict[str, object]]) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(self.file_path, mode="a", encoding=self.encoding) as file:
            for record in records:
                await file.write(self._serialize_line(record))

    async def save(self, data: dict) -> None:
        record = normalize_crawl_record(data)
        async with self._lock:
            self._buffer.append(record)
            if len(self._buffer) >= self.buffer_size:
                await self._flush_locked()

    async def _flush_locked(self) -> None:
        if not self._buffer:
            return
        batch = list(self._buffer)
        self._buffer.clear()
        try:
            await self.retry_strategy.execute_with_retry(self._write_lines, batch)
            self._stats["saved_count"] += len(batch)
            self._stats["flush_count"] += 1
        except Exception:
            self._stats["errors"] += 1
            self._buffer[0:0] = batch
            raise

    async def flush(self) -> None:
        async with self._lock:
            await self._flush_locked()

    async def close(self) -> None:
        await self.flush()

    def get_stats(self) -> dict[str, object]:
        stats = dict(self._stats)
        stats["buffered_count"] = len(self._buffer)
        stats["retry"] = self.retry_strategy.get_stats()
        return stats


class CSVStorage(DataStorage):
    def __init__(
        self,
        file_path: str,
        encoding: str = "utf-8",
        buffer_size: int = 1,
        retry_strategy: RetryStrategy | None = None,
    ) -> None:
        if not isinstance(file_path, str) or not file_path.strip():
            raise InvalidOperationError("file_path must be a non-empty string")
        if not isinstance(encoding, str) or not encoding.strip():
            raise InvalidOperationError("encoding must be a non-empty string")
        if not isinstance(buffer_size, int) or buffer_size < 1:
            raise InvalidOperationError("buffer_size must be a positive integer")
        if retry_strategy is not None and not isinstance(retry_strategy, RetryStrategy):
            raise InvalidOperationError("retry_strategy must be an instance of RetryStrategy")
        self.file_path = Path(file_path).expanduser()
        self.encoding = encoding.strip()
        self.buffer_size = buffer_size
        self.retry_strategy = retry_strategy if retry_strategy is not None else RetryStrategy(max_retries=1, base_delay=0.1)
        self._buffer: list[dict[str, object]] = []
        self._headers: list[str] | None = None
        self._lock = asyncio.Lock()
        self._stats = {"saved_count": 0, "flush_count": 0, "errors": 0, "path": str(self.file_path)}

    @staticmethod
    def _serialize_record(record: dict[str, object]) -> dict[str, object]:
        return {
            "url": record["url"],
            "title": record["title"],
            "text": record["text"],
            "links": json.dumps(list(record["links"]), ensure_ascii=False),
            "metadata": json.dumps(dict(record["metadata"]), ensure_ascii=False, sort_keys=True),
            "crawled_at": record["crawled_at"],
            "status_code": "" if record["status_code"] is None else str(record["status_code"]),
            "content_type": record["content_type"],
        }

    async def _write_rows(self, rows: list[dict[str, object]], headers: list[str], write_header: bool) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=headers, extrasaction="ignore", quoting=csv.QUOTE_MINIMAL)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        async with aiofiles.open(self.file_path, mode="a", encoding=self.encoding, newline="") as file:
            await file.write(buffer.getvalue())

    async def save(self, data: dict) -> None:
        record = self._serialize_record(normalize_crawl_record(data))
        async with self._lock:
            if self._headers is None:
                self._headers = list(record.keys())
            self._buffer.append(record)
            if len(self._buffer) >= self.buffer_size:
                await self._flush_locked()

    async def _flush_locked(self) -> None:
        if not self._buffer:
            return
        batch = list(self._buffer)
        self._buffer.clear()
        write_header = not self.file_path.exists() or self.file_path.stat().st_size == 0
        try:
            await self.retry_strategy.execute_with_retry(self._write_rows, batch, list(self._headers or []), write_header)
            self._stats["saved_count"] += len(batch)
            self._stats["flush_count"] += 1
        except Exception:
            self._stats["errors"] += 1
            self._buffer[0:0] = batch
            raise

    async def flush(self) -> None:
        async with self._lock:
            await self._flush_locked()

    async def close(self) -> None:
        await self.flush()

    def get_stats(self) -> dict[str, object]:
        stats = dict(self._stats)
        stats["buffered_count"] = len(self._buffer)
        stats["headers"] = list(self._headers or [])
        stats["retry"] = self.retry_strategy.get_stats()
        return stats


class SQLiteStorage(DataStorage):
    def __init__(
        self,
        db_path: str,
        table_name: str = "pages",
        batch_size: int = 20,
        retry_strategy: RetryStrategy | None = None,
    ) -> None:
        if not isinstance(db_path, str) or not db_path.strip():
            raise InvalidOperationError("db_path must be a non-empty string")
        if not isinstance(table_name, str) or not table_name.strip():
            raise InvalidOperationError("table_name must be a non-empty string")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name.strip()):
            raise InvalidOperationError("table_name contains invalid characters")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise InvalidOperationError("batch_size must be a positive integer")
        if retry_strategy is not None and not isinstance(retry_strategy, RetryStrategy):
            raise InvalidOperationError("retry_strategy must be an instance of RetryStrategy")
        self.db_path = Path(db_path).expanduser()
        self.table_name = table_name.strip()
        self.batch_size = batch_size
        self.retry_strategy = retry_strategy if retry_strategy is not None else RetryStrategy(max_retries=1, base_delay=0.1)
        self._buffer: list[dict[str, object]] = []
        self._connection: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        self._initialized = False
        self._stats = {"saved_count": 0, "flush_count": 0, "errors": 0, "path": str(self.db_path)}

    async def _init_db_unlocked(self) -> None:
        if self._initialized:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = await aiosqlite.connect(self.db_path)
        await self._connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                url TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                links_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                crawled_at TEXT NOT NULL,
                status_code INTEGER,
                content_type TEXT NOT NULL
            )
            """
        )
        await self._connection.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_crawled_at ON {self.table_name} (crawled_at)"
        )
        await self._connection.commit()
        self._initialized = True

    async def init_db(self) -> None:
        async with self._lock:
            await self._init_db_unlocked()

    @staticmethod
    def _to_row(record: dict[str, object]) -> tuple[object, ...]:
        return (
            record["url"],
            record["title"],
            record["text"],
            json.dumps(list(record["links"]), ensure_ascii=False),
            json.dumps(dict(record["metadata"]), ensure_ascii=False, sort_keys=True),
            record["crawled_at"],
            record["status_code"],
            record["content_type"],
        )

    async def _insert_rows(self, rows: list[tuple[object, ...]]) -> None:
        if not self._initialized:
            await self._init_db_unlocked()
        if self._connection is None:
            raise RuntimeError("database is not initialized")
        await self._connection.executemany(
            f"""
            INSERT OR REPLACE INTO {self.table_name}
            (url, title, text, links_json, metadata_json, crawled_at, status_code, content_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        await self._connection.commit()

    async def save(self, data: dict) -> None:
        record = normalize_crawl_record(data)
        async with self._lock:
            self._buffer.append(record)
            if len(self._buffer) >= self.batch_size:
                await self._flush_locked()

    async def _flush_locked(self) -> None:
        if not self._buffer:
            return
        batch = list(self._buffer)
        self._buffer.clear()
        rows = [self._to_row(record) for record in batch]
        try:
            await self.retry_strategy.execute_with_retry(self._insert_rows, rows)
            self._stats["saved_count"] += len(batch)
            self._stats["flush_count"] += 1
        except Exception:
            self._stats["errors"] += 1
            self._buffer[0:0] = batch
            raise

    async def flush(self) -> None:
        async with self._lock:
            await self._flush_locked()

    async def close(self) -> None:
        await self.flush()
        async with self._lock:
            if self._connection is not None:
                await self._connection.close()
                self._connection = None
            self._initialized = False

    def get_stats(self) -> dict[str, object]:
        stats = dict(self._stats)
        stats["buffered_count"] = len(self._buffer)
        stats["table_name"] = self.table_name
        stats["retry"] = self.retry_strategy.get_stats()
        return stats


class RetryStrategy:
    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        retry_on: list | None = None,
        base_delay: float = 0.5,
        max_delay: float = 30.0,
        timeout_backoff_factor: float = 1.5,
        max_retries_by_error: dict[type[Exception], int] | None = None,
    ) -> None:
        if not isinstance(max_retries, int) or max_retries < 0:
            raise InvalidOperationError("max_retries must be a non-negative integer")
        if not isinstance(backoff_factor, (int, float)) or backoff_factor < 1:
            raise InvalidOperationError("backoff_factor must be greater than or equal to 1")
        if not isinstance(base_delay, (int, float)) or base_delay < 0:
            raise InvalidOperationError("base_delay must be a non-negative number")
        if not isinstance(max_delay, (int, float)) or max_delay < 0:
            raise InvalidOperationError("max_delay must be a non-negative number")
        if not isinstance(timeout_backoff_factor, (int, float)) or timeout_backoff_factor < 1:
            raise InvalidOperationError("timeout_backoff_factor must be greater than or equal to 1")
        if retry_on is None:
            retry_on = [TransientError, NetworkError]
        if not isinstance(retry_on, list) or not all(isinstance(item, type) and issubclass(item, Exception) for item in retry_on):
            raise InvalidOperationError("retry_on must be a list of exception classes")
        if max_retries_by_error is not None:
            if not isinstance(max_retries_by_error, dict):
                raise InvalidOperationError("max_retries_by_error must be a dictionary")
            for error_type, retries in max_retries_by_error.items():
                if not isinstance(error_type, type) or not issubclass(error_type, Exception):
                    raise InvalidOperationError("max_retries_by_error keys must be exception classes")
                if not isinstance(retries, int) or retries < 0:
                    raise InvalidOperationError("max_retries_by_error values must be non-negative integers")
        self.max_retries = max_retries
        self.backoff_factor = float(backoff_factor)
        self.retry_on = tuple(retry_on)
        self.base_delay = float(base_delay)
        self.max_delay = float(max_delay)
        self.timeout_backoff_factor = float(timeout_backoff_factor)
        self.max_retries_by_error = dict(max_retries_by_error or {})
        self.reset_stats()

    @staticmethod
    def _extract_url(args: tuple[object, ...], kwargs: dict[str, object]) -> str:
        if isinstance(kwargs.get("url"), str):
            return str(kwargs["url"])
        if args and isinstance(args[0], str):
            return str(args[0])
        return ""

    def classify_error(self, error: Exception, url: str = "") -> CrawlerError:
        if isinstance(error, CrawlerError):
            if not error.url and url:
                error.url = url
            return error
        if isinstance(error, asyncio.TimeoutError):
            return TransientError("request timeout", url=url, original_error=error)
        if isinstance(error, aiohttp.ClientResponseError):
            message = f"HTTP {error.status}: {error.message}"
            if error.status == 429 or 500 <= error.status < 600:
                return TransientError(message, url=url, original_error=error, status_code=error.status)
            if error.status in (401, 403, 404) or 400 <= error.status < 500:
                return PermanentError(message, url=url, original_error=error, status_code=error.status)
        if isinstance(error, aiohttp.ClientError):
            return NetworkError(str(error), url=url, original_error=error)
        if isinstance(error, OSError):
            return NetworkError(str(error), url=url, original_error=error)
        return PermanentError(str(error), url=url, original_error=error)

    def record_error(self, error: Exception, url: str = "") -> CrawlerError:
        classified_error = self.classify_error(error, url=url)
        error_type = type(classified_error).__name__
        self._error_counts_by_type[error_type] = self._error_counts_by_type.get(error_type, 0) + 1
        if isinstance(classified_error, PermanentError) and classified_error.url:
            self._permanent_error_urls.add(classified_error.url)
        return classified_error

    def _resolve_max_retries(self, error: Exception) -> int:
        for error_type, retries in self.max_retries_by_error.items():
            if isinstance(error, error_type):
                return retries
        return self.max_retries

    def _should_retry(self, error: Exception) -> bool:
        return isinstance(error, self.retry_on)

    def _get_delay_multiplier(self, error: Exception) -> float:
        if isinstance(error, NetworkError):
            return 0.5
        if isinstance(error, TransientError) and getattr(error, "status_code", None) == 429:
            return 2.0
        return 1.0

    def _get_delay(self, error: Exception, attempt_number: int) -> float:
        delay = self.base_delay * self._get_delay_multiplier(error) * (self.backoff_factor ** max(0, attempt_number - 1))
        return min(self.max_delay, delay)

    def get_timeout_multiplier(self, attempt_number: int) -> float:
        return self.timeout_backoff_factor ** max(0, attempt_number - 1)

    async def _invoke(self, coro, args: tuple[object, ...], kwargs: dict[str, object], attempt_number: int, last_error: Exception | None):
        call_kwargs = dict(kwargs)
        try:
            signature = inspect.signature(coro)
            supports_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
            if supports_kwargs or "_retry_attempt" in signature.parameters:
                call_kwargs["_retry_attempt"] = attempt_number
            if supports_kwargs or "_retry_error" in signature.parameters:
                call_kwargs["_retry_error"] = last_error
        except (TypeError, ValueError):
            pass
        return await coro(*args, **call_kwargs)

    async def execute_with_retry(self, coro, *args, **kwargs):
        url = self._extract_url(args, kwargs)
        attempt_number = 1
        last_error: Exception | None = None
        while True:
            try:
                result = await self._invoke(coro, args, kwargs, attempt_number, last_error)
                if attempt_number > 1:
                    self._successful_retries += 1
                return result
            except Exception as error:
                classified_error = self.record_error(error, url=url)
                max_retries_for_error = self._resolve_max_retries(classified_error)
                should_retry = self._should_retry(classified_error) and attempt_number - 1 < max_retries_for_error
                if not should_retry:
                    logger.warning(
                        "Retry finished with failure for %s on attempt %s: %s (%s)",
                        url or "<unknown>",
                        attempt_number,
                        classified_error,
                        type(classified_error).__name__,
                    )
                    raise classified_error
                delay = self._get_delay(classified_error, attempt_number)
                self._retry_attempts_total += 1
                self._total_retry_delay += delay
                logger.info(
                    "Retry scheduled for %s on attempt %s/%s after %.2fs due to %s (%s)",
                    url or "<unknown>",
                    attempt_number,
                    max_retries_for_error + 1,
                    delay,
                    classified_error,
                    type(classified_error).__name__,
                )
                await asyncio.sleep(delay)
                last_error = classified_error
                attempt_number += 1

    def get_stats(self) -> dict[str, object]:
        return {
            "max_retries": self.max_retries,
            "backoff_factor": self.backoff_factor,
            "retry_attempts_total": self._retry_attempts_total,
            "successful_retries": self._successful_retries,
            "average_retry_delay": self._total_retry_delay / self._retry_attempts_total if self._retry_attempts_total else 0.0,
            "error_counts_by_type": dict(self._error_counts_by_type),
            "permanent_error_urls": sorted(self._permanent_error_urls),
        }

    def reset_stats(self) -> None:
        self._retry_attempts_total = 0
        self._successful_retries = 0
        self._total_retry_delay = 0.0
        self._error_counts_by_type: dict[str, int] = {}
        self._permanent_error_urls: set[str] = set()


class AsyncCrawler:
    def __init__(
        self,
        max_concurrent: int = 10,
        connect_timeout: float = 5,
        read_timeout: float = 10,
        total_timeout: float = 15,
        html_parser: HTMLParser | None = None,
        max_depth: int = 2,
        per_domain_concurrent: int | None = None,
        requests_per_second: float = 1.0,
        rate_limit_per_domain: bool = True,
        respect_robots: bool = True,
        min_delay: float = 0.0,
        jitter: float = 0.0,
        user_agent: str = "AsyncCrawler/1.0",
        user_agents: list[str] | None = None,
        backoff_base: float = 0.5,
        backoff_factor: float = 2.0,
        backoff_max: float = 30.0,
        retry_strategy: RetryStrategy | None = None,
        storage: DataStorage | None = None,
    ) -> None:
        if not isinstance(max_concurrent, int) or max_concurrent < 1:
            raise InvalidOperationError("max_concurrent must be a positive integer")
        if not isinstance(max_depth, int) or max_depth < 0:
            raise InvalidOperationError("max_depth must be a non-negative integer")
        if per_domain_concurrent is None:
            per_domain_concurrent = max_concurrent
        if not isinstance(per_domain_concurrent, int) or per_domain_concurrent < 1:
            raise InvalidOperationError("per_domain_concurrent must be a positive integer")
        if not isinstance(requests_per_second, (int, float)) or requests_per_second <= 0:
            raise InvalidOperationError("requests_per_second must be a positive number")
        if not isinstance(rate_limit_per_domain, bool):
            raise InvalidOperationError("rate_limit_per_domain must be a boolean")
        if not isinstance(respect_robots, bool):
            raise InvalidOperationError("respect_robots must be a boolean")
        if not isinstance(min_delay, (int, float)) or min_delay < 0:
            raise InvalidOperationError("min_delay must be a non-negative number")
        if not isinstance(jitter, (int, float)) or jitter < 0:
            raise InvalidOperationError("jitter must be a non-negative number")
        if not isinstance(user_agent, str) or not user_agent.strip():
            raise InvalidOperationError("user_agent must be a non-empty string")
        if user_agents is not None:
            if not isinstance(user_agents, list) or not user_agents or not all(isinstance(item, str) and item.strip() for item in user_agents):
                raise InvalidOperationError("user_agents must be a non-empty list of non-empty strings")
        if not isinstance(backoff_base, (int, float)) or backoff_base < 0:
            raise InvalidOperationError("backoff_base must be a non-negative number")
        if not isinstance(backoff_factor, (int, float)) or backoff_factor < 1:
            raise InvalidOperationError("backoff_factor must be greater than or equal to 1")
        if not isinstance(backoff_max, (int, float)) or backoff_max < 0:
            raise InvalidOperationError("backoff_max must be a non-negative number")
        if retry_strategy is not None and not isinstance(retry_strategy, RetryStrategy):
            raise InvalidOperationError("retry_strategy must be an instance of RetryStrategy")
        if storage is not None and not isinstance(storage, DataStorage):
            raise InvalidOperationError("storage must be an instance of DataStorage")
        self.max_concurrent = max_concurrent
        self.max_depth = max_depth
        self.per_domain_concurrent = per_domain_concurrent
        self.connect_timeout = float(connect_timeout)
        self.read_timeout = float(read_timeout)
        self.total_timeout = float(total_timeout)
        self.requests_per_second = float(requests_per_second)
        self.rate_limit_per_domain = rate_limit_per_domain
        self.respect_robots = respect_robots
        self.min_delay = float(min_delay)
        self.jitter = float(jitter)
        self.user_agent = user_agent.strip()
        self.user_agents = [item.strip() for item in user_agents] if user_agents is not None else None
        self.backoff_base = float(backoff_base)
        self.backoff_factor = float(backoff_factor)
        self.backoff_max = float(backoff_max)
        self._timeout = aiohttp.ClientTimeout(
            total=total_timeout,
            connect=connect_timeout,
            sock_read=read_timeout,
        )
        self._session: aiohttp.ClientSession | None = None
        self.html_parser = html_parser if html_parser is not None else HTMLParser()
        self.semaphore_manager = SemaphoreManager(max_concurrent=max_concurrent, per_domain_concurrent=per_domain_concurrent)
        self.rate_limiter = RateLimiter(requests_per_second=self.requests_per_second, per_domain=self.rate_limit_per_domain)
        self.robots_parser = RobotsParser(timeout=connect_timeout, user_agent=self.user_agent)
        self.retry_strategy = retry_strategy if retry_strategy is not None else RetryStrategy()
        self.storage = storage
        self.visited_urls: set[str] = set()
        self.failed_urls: dict[str, str] = {}
        self.processed_urls: dict[str, dict[str, object]] = {}
        self.url_depths: dict[str, int] = {}
        self.blocked_urls: dict[str, str] = {}
        self.error_details: dict[str, dict[str, object]] = {}
        self.storage_errors: dict[str, str] = {}
        self._response_metadata: dict[str, dict[str, object]] = {}
        self._crawl_started_at = 0.0
        self._request_delay_total = 0.0
        self._request_delay_count = 0
        self._request_started_at: deque[float] = deque()
        self._domain_backoffs: dict[str, float] = {}
        self._backoff_count = 0
        self._user_agent_index = 0
        self._user_agent_lock = asyncio.Lock()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=self.max_concurrent,
                limit_per_host=self.per_domain_concurrent,
                ttl_dns_cache=300,
            )
            self._session = aiohttp.ClientSession(
                timeout=self._timeout,
                connector=connector,
                headers={"User-Agent": self.user_agent},
            )
        return self._session

    @staticmethod
    def _is_supported_url(url: str) -> bool:
        return isinstance(url, str) and url.startswith(("http://", "https://"))

    @staticmethod
    def _normalize_crawl_url(url: str) -> str:
        if not isinstance(url, str):
            return ""
        normalized_url, _fragment = urldefrag(url.strip())
        return normalized_url

    @staticmethod
    def _matches_patterns(url: str, patterns: list[str] | None) -> bool:
        if not patterns:
            return False
        return any(re.search(pattern, url) for pattern in patterns)

    def _should_include_url(
        self,
        url: str,
        allowed_domains: set[str],
        same_domain_only: bool,
        include_patterns: list[str] | None,
        exclude_patterns: list[str] | None,
    ) -> bool:
        if not self._is_supported_url(url):
            return False
        if same_domain_only and urlparse(url).netloc not in allowed_domains:
            return False
        if exclude_patterns and self._matches_patterns(url, exclude_patterns):
            return False
        if include_patterns and not self._matches_patterns(url, include_patterns):
            return False
        return True

    def _reset_crawl_state(self) -> None:
        self.visited_urls.clear()
        self.failed_urls.clear()
        self.processed_urls.clear()
        self.url_depths.clear()
        self.blocked_urls.clear()
        self.error_details.clear()
        self.storage_errors.clear()
        self._response_metadata.clear()
        self.retry_strategy.reset_stats()
        self._crawl_started_at = perf_counter()
        self._request_delay_total = 0.0
        self._request_delay_count = 0
        self._request_started_at.clear()
        self._domain_backoffs.clear()
        self._backoff_count = 0

    def _get_crawl_stats(self, queue: CrawlerQueue) -> dict[str, object]:
        elapsed = perf_counter() - self._crawl_started_at if self._crawl_started_at else 0.0
        now = perf_counter()
        while self._request_started_at and self._request_started_at[0] < now - 1.0:
            self._request_started_at.popleft()
        processed_count = len(self.processed_urls)
        failed_count = len(self.failed_urls)
        total_finished = processed_count + failed_count
        pages_per_second = total_finished / elapsed if elapsed > 0 else 0.0
        stats = queue.get_stats()
        stats.update(
            {
                "visited": len(self.visited_urls),
                "elapsed": elapsed,
                "pages_per_second": pages_per_second,
                "processed_pages": processed_count,
                "failed_pages": failed_count,
                "robots_blocked": len(self.blocked_urls),
                "average_delay": self._request_delay_total / self._request_delay_count if self._request_delay_count else 0.0,
                "current_req_per_sec": float(len(self._request_started_at)),
                "backoff_count": self._backoff_count,
            }
        )
        stats["semaphores"] = self.semaphore_manager.get_stats()
        stats["rate_limiter"] = self.rate_limiter.get_stats()
        stats["robots"] = self.robots_parser.get_stats()
        stats["retry"] = self.retry_strategy.get_stats()
        stats["storage_errors"] = len(self.storage_errors)
        stats["storage"] = self.storage.get_stats() if self.storage is not None else {}
        return stats

    def _log_crawl_progress(self, queue: CrawlerQueue) -> None:
        stats = self._get_crawl_stats(queue)
        logger.info(
            "Crawl progress: processed=%s queued=%s errors=%s blocked=%s speed=%.2f pages/sec req_rate=%.2f avg_delay=%.2f",
            stats["processed_pages"],
            stats["pending"],
            stats["failed_pages"],
            stats["robots_blocked"],
            float(stats["pages_per_second"]),
            float(stats["current_req_per_sec"]),
            float(stats["average_delay"]),
        )

    @staticmethod
    def _get_base_url(url: str) -> str:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return ""
        return f"{parsed.scheme}://{parsed.netloc}"

    async def _get_request_user_agent(self) -> str:
        if not self.user_agents:
            return self.user_agent
        async with self._user_agent_lock:
            user_agent = self.user_agents[self._user_agent_index % len(self.user_agents)]
            self._user_agent_index += 1
            return user_agent

    def _record_request_delay(self, delay: float) -> None:
        self._request_delay_total += delay
        self._request_delay_count += 1

    def _record_request_start(self) -> None:
        started_at = perf_counter()
        self._request_started_at.append(started_at)
        while self._request_started_at and self._request_started_at[0] < started_at - 1.0:
            self._request_started_at.popleft()

    def _should_backoff(self, error: Exception | None) -> bool:
        if isinstance(error, aiohttp.ClientResponseError):
            return error.status == 429 or error.status >= 500
        return isinstance(error, (aiohttp.ClientError, asyncio.TimeoutError))

    def _increase_backoff(self, domain: str) -> None:
        if not domain:
            return
        current_delay = self._domain_backoffs.get(domain, 0.0)
        next_delay = self.backoff_base if current_delay <= 0 else min(self.backoff_max, current_delay * self.backoff_factor)
        self._domain_backoffs[domain] = next_delay
        self._backoff_count += 1

    def _reset_backoff(self, domain: str) -> None:
        if domain:
            self._domain_backoffs[domain] = 0.0

    def _build_storage_record(self, url: str, parsed_result: dict[str, object]) -> dict[str, object]:
        response_metadata = dict(self._response_metadata.get(url, {}))
        return normalize_crawl_record(
            {
                "url": url,
                "title": parsed_result.get("title", ""),
                "text": parsed_result.get("text", ""),
                "links": list(parsed_result.get("links", [])),
                "metadata": dict(parsed_result.get("metadata", {})),
                "crawled_at": _utc_now(),
                "status_code": response_metadata.get("status_code"),
                "content_type": response_metadata.get("content_type", ""),
            }
        )

    async def _save_processed_page(self, url: str, parsed_result: dict[str, object]) -> None:
        if self.storage is None:
            return
        record = self._build_storage_record(url, parsed_result)
        try:
            await self.storage.save(record)
        except Exception as error:
            self.storage_errors[url] = str(error)
            logger.warning("Storage error for %s: %s", url, error)

    def _get_attempt_timeout(self, attempt_number: int) -> aiohttp.ClientTimeout:
        multiplier = self.retry_strategy.get_timeout_multiplier(attempt_number)
        return aiohttp.ClientTimeout(
            total=self.total_timeout * multiplier,
            connect=self.connect_timeout * multiplier,
            sock_read=self.read_timeout * multiplier,
        )

    async def _apply_politeness(self, url: str, user_agent: str) -> str | None:
        domain = urlparse(url).netloc
        base_url = self._get_base_url(url)
        crawl_delay = 0.0
        if self.respect_robots:
            await self.robots_parser.fetch_robots(base_url)
            if not self.robots_parser.can_fetch(url, user_agent=user_agent):
                logger.info("Blocked by robots.txt: %s", url)
                self.blocked_urls[url] = "blocked by robots.txt"
                return "blocked by robots.txt"
            crawl_delay = self.robots_parser._get_crawl_delay_for_domain(base_url, user_agent=user_agent)
        rate_limit_key = domain if self.rate_limit_per_domain else None
        limiter_wait = await self.rate_limiter.acquire(rate_limit_key)
        base_delay = max(self.min_delay, crawl_delay, self._domain_backoffs.get(domain, 0.0))
        jitter_delay = random.uniform(0.0, self.jitter) if self.jitter > 0 else 0.0
        extra_delay = max(0.0, base_delay - limiter_wait) + jitter_delay
        total_delay = limiter_wait + extra_delay
        if extra_delay > 0:
            await asyncio.sleep(extra_delay)
        if total_delay > 0:
            self._record_request_delay(total_delay)
        return None

    async def _fetch_url_once(
        self,
        url: str,
        request_user_agent: str,
        _retry_attempt: int = 1,
        _retry_error: Exception | None = None,
    ) -> str:
        if not self._is_supported_url(url):
            raise PermanentError("unsupported url", url=url)

        async with self.semaphore_manager.limit(url):
            polite_error = await self._apply_politeness(url, request_user_agent)
            if polite_error is not None:
                raise PermanentError(polite_error, url=url)
            logger.info("Starting download attempt %s: %s", _retry_attempt, url)
            self._record_request_start()
            domain = urlparse(url).netloc
            try:
                session = await self._ensure_session()
                async with session.get(
                    url,
                    headers={"User-Agent": request_user_agent},
                    timeout=self._get_attempt_timeout(_retry_attempt),
                ) as response:
                    response.raise_for_status()
                    self._response_metadata[url] = {
                        "status_code": response.status,
                        "content_type": response.headers.get("Content-Type", ""),
                    }
                    content = await response.text()
                    self._reset_backoff(domain)
                    logger.info("Completed download on attempt %s: %s", _retry_attempt, url)
                    return content
            except aiohttp.ClientResponseError as error:
                if self._should_backoff(error):
                    self._increase_backoff(domain)
                logger.warning("HTTP error for %s on attempt %s: %s", url, _retry_attempt, error)
                raise self.retry_strategy.classify_error(error, url=url)
            except asyncio.TimeoutError as error:
                if self._should_backoff(error):
                    self._increase_backoff(domain)
                logger.warning("Timeout error for %s on attempt %s: %s", url, _retry_attempt, error)
                raise self.retry_strategy.classify_error(error, url=url)
            except aiohttp.ClientError as error:
                if self._should_backoff(error):
                    self._increase_backoff(domain)
                logger.warning("Network error for %s on attempt %s: %s", url, _retry_attempt, error)
                raise self.retry_strategy.classify_error(error, url=url)

    async def _fetch_url_with_error(self, url: str) -> tuple[str, str | None]:
        if not self._is_supported_url(url):
            error = "unsupported url"
            logger.warning("Unsupported URL skipped: %s", url)
            return "", error
        request_user_agent = await self._get_request_user_agent()
        try:
            content = await self.retry_strategy.execute_with_retry(self._fetch_url_once, url, request_user_agent)
            return content, None
        except CrawlerError as error:
            if isinstance(error, PermanentError) and str(error) == "blocked by robots.txt":
                self.blocked_urls[url] = str(error)
            self.error_details[url] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            return "", str(error)

    async def fetch_url(self, url: str) -> str:
        content, _error = await self._fetch_url_with_error(url)
        return content

    async def fetch_urls(self, urls: list[str]) -> dict[str, str]:
        tasks = [self.fetch_url(url) for url in urls]
        results = await asyncio.gather(*tasks)
        return dict(zip(urls, results))

    async def fetch_and_parse(self, url: str) -> dict[str, object]:
        html = await self.fetch_url(url)
        if not html:
            return self.html_parser.empty_result(url)
        try:
            parsed_result = await self.html_parser.parse_html(html, url)
            await self._save_processed_page(url, parsed_result)
            return parsed_result
        except Exception as error:
            parse_error = self.retry_strategy.record_error(ParseError(str(error), url=url, original_error=error), url=url)
            self.error_details[url] = {
                "type": type(parse_error).__name__,
                "message": str(parse_error),
            }
            logger.warning("Parse error for %s: %s", url, parse_error)
            return self.html_parser.empty_result(url)

    async def _enqueue_url(
        self,
        queue: CrawlerQueue,
        url: str,
        depth: int,
        allowed_domains: set[str],
        same_domain_only: bool,
        include_patterns: list[str] | None,
        exclude_patterns: list[str] | None,
    ) -> bool:
        normalized_url = self._normalize_crawl_url(url)
        if not normalized_url:
            return False
        if depth > self.max_depth:
            return False
        if normalized_url in self.visited_urls:
            return False
        if normalized_url in self.processed_urls:
            return False
        if normalized_url in self.failed_urls:
            return False
        if normalized_url in self.blocked_urls:
            return False
        if queue.is_known(normalized_url):
            return False
        if not self._should_include_url(
            normalized_url,
            allowed_domains=allowed_domains,
            same_domain_only=same_domain_only,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        ):
            return False
        self.url_depths[normalized_url] = depth
        return await queue.add_url(normalized_url, priority=depth, depth=depth)

    async def _process_crawl_url(
        self,
        queue: CrawlerQueue,
        url: str,
        depth: int,
        max_pages: int,
        allowed_domains: set[str],
        same_domain_only: bool,
        include_patterns: list[str] | None,
        exclude_patterns: list[str] | None,
    ) -> None:
        content, error = await self._fetch_url_with_error(url)
        if error or not content:
            failure_reason = error or "empty response"
            queue.mark_failed(url, failure_reason)
            if failure_reason == "blocked by robots.txt":
                self.blocked_urls[url] = failure_reason
            self.failed_urls[url] = failure_reason
            self._log_crawl_progress(queue)
            return

        try:
            parsed_result = await self.html_parser.parse_html(content, url)
        except Exception as error:
            parse_error = self.retry_strategy.record_error(ParseError(str(error), url=url, original_error=error), url=url)
            queue.mark_failed(url, str(parse_error))
            self.failed_urls[url] = str(parse_error)
            self.error_details[url] = {
                "type": type(parse_error).__name__,
                "message": str(parse_error),
            }
            self._log_crawl_progress(queue)
            return
        parsed_result["depth"] = depth
        self.processed_urls[url] = parsed_result
        await self._save_processed_page(url, parsed_result)
        queue.mark_processed(url)

        if depth < self.max_depth:
            for link in list(parsed_result.get("links", [])):
                if len(self.processed_urls) + len(self.failed_urls) + queue.pending_count() >= max_pages:
                    break
                await self._enqueue_url(
                    queue,
                    link,
                    depth=depth + 1,
                    allowed_domains=allowed_domains,
                    same_domain_only=same_domain_only,
                    include_patterns=include_patterns,
                    exclude_patterns=exclude_patterns,
                )
        self._log_crawl_progress(queue)

    async def crawl(
        self,
        start_urls: list[str],
        max_pages: int = 100,
        same_domain_only: bool = False,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        max_depth: int | None = None,
    ) -> dict[str, object]:
        if not isinstance(start_urls, list) or not start_urls:
            raise InvalidOperationError("start_urls must be a non-empty list")
        if not isinstance(max_pages, int) or max_pages < 1:
            raise InvalidOperationError("max_pages must be a positive integer")
        if max_depth is not None:
            if not isinstance(max_depth, int) or max_depth < 0:
                raise InvalidOperationError("max_depth must be a non-negative integer")
            self.max_depth = max_depth

        self._reset_crawl_state()
        queue = CrawlerQueue()
        normalized_start_urls = [self._normalize_crawl_url(url) for url in start_urls]
        allowed_domains = {urlparse(url).netloc for url in normalized_start_urls if self._is_supported_url(url)}

        for start_url in normalized_start_urls:
            await self._enqueue_url(
                queue,
                start_url,
                depth=0,
                allowed_domains=allowed_domains,
                same_domain_only=False,
                include_patterns=None,
                exclude_patterns=None,
            )

        tasks: set[asyncio.Task[None]] = set()
        try:
            while len(self.processed_urls) + len(self.failed_urls) < max_pages and (queue.pending_count() > 0 or tasks):
                while queue.pending_count() > 0 and len(tasks) < self.max_concurrent and len(self.visited_urls) < max_pages:
                    next_url = await queue.get_next()
                    if next_url is None:
                        break
                    depth = queue.get_depth(next_url)
                    if next_url in self.visited_urls:
                        queue.mark_processed(next_url)
                        continue
                    if depth > self.max_depth:
                        queue.mark_processed(next_url)
                        continue
                    self.visited_urls.add(next_url)
                    self.url_depths[next_url] = depth
                    tasks.add(
                        asyncio.create_task(
                            self._process_crawl_url(
                                queue,
                                next_url,
                                depth=depth,
                                max_pages=max_pages,
                                allowed_domains=allowed_domains,
                                same_domain_only=same_domain_only,
                                include_patterns=include_patterns,
                                exclude_patterns=exclude_patterns,
                            )
                        )
                    )

                if not tasks:
                    break

                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                tasks = set(pending)
                for completed_task in done:
                    await completed_task
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        return {
            "processed_urls": dict(self.processed_urls),
            "failed_urls": dict(self.failed_urls),
            "blocked_urls": dict(self.blocked_urls),
            "error_details": dict(self.error_details),
            "storage_errors": dict(self.storage_errors),
            "visited_urls": sorted(self.visited_urls),
            "stats": self._get_crawl_stats(queue),
        }

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None
        if self.storage is not None:
            try:
                await self.storage.close()
            except Exception as error:
                logger.warning("Storage close error: %s", error)

    async def __aenter__(self) -> AsyncCrawler:
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()
