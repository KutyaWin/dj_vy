from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from uuid import uuid4


TWOPLACES = Decimal("0.01")
INVESTMENT_ASSET_TYPES = ("stocks", "bonds", "etf")


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


class Bank:
    def __init__(self, name: str = "Bank") -> None:
        self.name = Owner._validate_text(name, "name")
        self.clients: dict[str, Client] = {}
        self.accounts: dict[str, AbstractAccount] = {}
        self.account_to_client: dict[str, str] = {}

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

    def _mark_suspicious(self, client: Client, event: str) -> None:
        client.add_suspicious_activity(event)

    def _current_hour(self) -> int:
        return datetime.now().hour

    def _ensure_operation_time_allowed(self, client: Client, action_name: str) -> None:
        current_hour = self._current_hour()
        if 0 <= current_hour < 5:
            self._mark_suspicious(client, f"restricted hours operation: {action_name}")
            raise InvalidOperationError("operations are not allowed from 00:00 to 05:00")

    def add_client(self, client: Client) -> Client:
        if not isinstance(client, Client):
            raise InvalidOperationError("client must be a Client instance")
        if client.client_id in self.clients:
            raise InvalidOperationError("client_id must be unique")
        self.clients[client.client_id] = client
        return client

    def open_account(self, client_id: str, account_type: str = "bank", **kwargs: object) -> AbstractAccount:
        client = self._get_client(client_id)
        self._ensure_operation_time_allowed(client, "open_account")
        if client.is_locked:
            self._mark_suspicious(client, "blocked client attempted to open account")
            raise InvalidOperationError("blocked client cannot open account")
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
        account_kwargs.setdefault("account_id", self._generate_unique_account_id())
        account = account_classes[normalized_account_type](**account_kwargs)
        self.accounts[account.account_id] = account
        self.account_to_client[account.account_id] = client.client_id
        client.add_account_id(account.account_id)
        return account

    def close_account(self, account_id: str) -> AccountStatus:
        account = self._get_account(account_id)
        client = self._get_account_owner_client(account_id)
        self._ensure_operation_time_allowed(client, "close_account")
        return account.close()

    def freeze_account(self, account_id: str) -> AccountStatus:
        account = self._get_account(account_id)
        client = self._get_account_owner_client(account_id)
        self._ensure_operation_time_allowed(client, "freeze_account")
        return account.freeze()

    def unfreeze_account(self, account_id: str) -> AccountStatus:
        account = self._get_account(account_id)
        client = self._get_account_owner_client(account_id)
        self._ensure_operation_time_allowed(client, "unfreeze_account")
        return account.activate()

    def authenticate_client(self, client_id: str, pin_code: str) -> bool:
        client = self._get_client(client_id)
        if client.is_locked:
            self._mark_suspicious(client, "authentication attempt on blocked client")
            raise InvalidOperationError("client is blocked")
        if client.pin_code != pin_code:
            client.failed_auth_attempts += 1
            if client.failed_auth_attempts >= 3:
                client.is_locked = True
                client.status = ClientStatus.BLOCKED
                client.add_suspicious_activity("client locked after 3 failed authentication attempts")
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

    def get_total_balance(self) -> Decimal:
        total_balance = Decimal("0.00")
        for account in self.accounts.values():
            total_balance += account.balance
            if isinstance(account, InvestmentAccount):
                total_balance += account._portfolio_total_value()
        return total_balance.quantize(TWOPLACES, rounding=ROUND_HALF_UP)

    def get_clients_ranking(self) -> list[dict[str, object]]:
        ranking: list[dict[str, object]] = []
        for client in self.clients.values():
            total_assets = Decimal("0.00")
            for account_id in client.account_ids:
                account = self.accounts[account_id]
                total_assets += account.balance
                if isinstance(account, InvestmentAccount):
                    total_assets += account._portfolio_total_value()
            ranking.append(
                {
                    "client_id": client.client_id,
                    "full_name": client.full_name,
                    "status": client.status.value,
                    "accounts_count": len(client.account_ids),
                    "total_assets": f"{total_assets.quantize(TWOPLACES, rounding=ROUND_HALF_UP):.2f}",
                }
            )
        ranking.sort(key=lambda item: Decimal(str(item["total_assets"])), reverse=True)
        return ranking
