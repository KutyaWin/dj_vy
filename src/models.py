from __future__ import annotations

from abc import ABC, abstractmethod
import csv
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
import json
from pathlib import Path
from uuid import uuid4


TWOPLACES = Decimal("0.01")
INVESTMENT_ASSET_TYPES = ("stocks", "bonds", "etf")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_utc_datetime(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise InvalidOperationError(f"{field_name} must be a datetime instance")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


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
        return (normalized_rate / Decimal("100")).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


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
    ) -> None:
        self.large_amount_threshold = Transaction._normalize_amount(large_amount_threshold)
        if not isinstance(frequent_operations_threshold, int) or frequent_operations_threshold <= 0:
            raise InvalidOperationError("frequent_operations_threshold must be a positive integer")
        if not isinstance(frequent_operations_window_minutes, int) or frequent_operations_window_minutes <= 0:
            raise InvalidOperationError("frequent_operations_window_minutes must be a positive integer")
        self.frequent_operations_threshold = frequent_operations_threshold
        self.frequent_operations_window_minutes = frequent_operations_window_minutes
        self.block_level = AuditEvent._coerce_risk_level(block_level)

    def _risk_from_score(self, score: int) -> RiskLevel:
        if score >= 4:
            return RiskLevel.HIGH
        if score >= 2:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _should_block(self, risk_level: RiskLevel) -> bool:
        risk_order = {RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2, RiskLevel.HIGH: 3}
        return risk_order[risk_level] >= risk_order[self.block_level]

    def analyze_transaction(self, transaction: Transaction, bank: Bank, current_time: datetime) -> RiskAssessment:
        if not isinstance(transaction, Transaction):
            raise InvalidOperationError("transaction must be a Transaction instance")
        if not isinstance(bank, Bank):
            raise InvalidOperationError("bank must be a Bank instance")
        normalized_current_time = _normalize_utc_datetime(current_time, "current_time")
        sender_client = bank._get_account_owner_client(transaction.sender_account_id)
        reasons: list[str] = []
        score = 0
        if transaction.amount >= self.large_amount_threshold:
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
            return datetime.now().hour
        return current_time.hour

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
        audit_events = [event.to_dict() for event in self.bank.audit_log.filter_events(client_id=client.client_id)]
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
        transaction.fee = self._calculate_fee(same_client)
        recipient_amount = self._convert_amount(transaction.amount, sender_account.currency, recipient_account.currency)
        total_sender_charge = (transaction.amount + transaction.fee).quantize(TWOPLACES, rounding=ROUND_HALF_UP)
        sender_account.withdraw(total_sender_charge)
        recipient_account.deposit(recipient_amount)
        return {
            "sender_currency": sender_account.currency.value,
            "recipient_currency": recipient_account.currency.value,
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
