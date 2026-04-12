
from decimal import Decimal
import unittest
from unittest.mock import patch

from src.models import (
    AccountClosedError,
    AccountFrozenError,
    AccountStatus,
    BankAccount,
    Bank,
    Client,
    ClientStatus,
    Currency,
    InvestmentAccount,
    InsufficientFundsError,
    InvalidOperationError,
    Owner,
    PremiumAccount,
    SavingsAccount,
)


class BankAccountTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.owner = Owner(
            full_name="Ivan Petrov",
            email="ivan.petrov@example.com",
            phone="+77001234567",
        )

    def test_creates_active_account_with_auto_generated_id(self) -> None:
        account = BankAccount(owner=self.owner, balance="1500", currency=Currency.KZT)

        self.assertEqual(account.status, AccountStatus.ACTIVE)
        self.assertEqual(account.balance, Decimal("1500.00"))
        self.assertEqual(account.currency, Currency.KZT)
        self.assertEqual(len(account.account_id), 8)

    def test_creates_frozen_account(self) -> None:
        account = BankAccount(
            owner=self.owner,
            balance="250.25",
            status=AccountStatus.FROZEN,
            currency="USD",
            account_id="ACC9988",
        )

        self.assertEqual(account.status, AccountStatus.FROZEN)
        self.assertEqual(account.balance, Decimal("250.25"))
        self.assertEqual(account.account_id, "ACC9988")

    def test_deposit_returns_updated_balance(self) -> None:
        account = BankAccount(owner=self.owner, balance="100.00")

        new_balance = account.deposit("49.995")

        self.assertEqual(new_balance, Decimal("150.00"))
        self.assertEqual(account.balance, Decimal("150.00"))

    def test_withdraw_returns_updated_balance(self) -> None:
        account = BankAccount(owner=self.owner, balance="200.00", currency="EUR")

        new_balance = account.withdraw("75.50")

        self.assertEqual(new_balance, Decimal("124.50"))
        self.assertEqual(account.balance, Decimal("124.50"))

    def test_withdraw_raises_when_funds_are_insufficient(self) -> None:
        account = BankAccount(owner=self.owner, balance="20.00")

        with self.assertRaises(InsufficientFundsError):
            account.withdraw("20.01")

    def test_frozen_account_blocks_operations(self) -> None:
        account = BankAccount(owner=self.owner, balance="500.00", status="frozen")

        with self.assertRaises(AccountFrozenError):
            account.deposit("1")

        with self.assertRaises(AccountFrozenError):
            account.withdraw("1")

    def test_closed_account_blocks_operations(self) -> None:
        account = BankAccount(owner=self.owner, balance="500.00")
        account.close()

        with self.assertRaises(AccountClosedError):
            account.deposit("10")

        with self.assertRaises(AccountClosedError):
            account.withdraw("10")

    def test_status_transition_methods(self) -> None:
        account = BankAccount(owner=self.owner, balance="300.00")

        self.assertEqual(account.freeze(), AccountStatus.FROZEN)
        self.assertEqual(account.activate(), AccountStatus.ACTIVE)
        self.assertEqual(account.close(), AccountStatus.CLOSED)

        with self.assertRaises(AccountClosedError):
            account.activate()

    def test_get_account_info_returns_masked_and_safe_data(self) -> None:
        account = BankAccount(
            owner=self.owner,
            balance="999.90",
            currency="USD",
            account_id="ABCD1234",
        )

        info = account.get_account_info()

        self.assertEqual(info["account_id"], "****1234")
        self.assertEqual(info["status"], "active")
        self.assertEqual(info["balance"], "999.90")
        self.assertEqual(info["currency"], "USD")
        self.assertEqual(info["owner"]["full_name"], "Ivan Petrov")
        self.assertNotEqual(info["owner"]["email"], "ivan.petrov@example.com")
        self.assertNotEqual(info["owner"]["phone"], "+77001234567")

    def test_str_contains_required_fields(self) -> None:
        account = BankAccount(
            owner=self.owner,
            balance="88.10",
            currency="RUB",
            account_id="ZXCV4321",
        )

        account_repr = str(account)

        self.assertIn("BankAccount", account_repr)
        self.assertIn("Ivan Petrov", account_repr)
        self.assertIn("****4321", account_repr)
        self.assertIn("active", account_repr)
        self.assertIn("88.10 RUB", account_repr)

    def test_invalid_amount_raises_error(self) -> None:
        account = BankAccount(owner=self.owner, balance="100.00")

        with self.assertRaises(InvalidOperationError):
            account.deposit("0")

        with self.assertRaises(InvalidOperationError):
            account.withdraw("-10")

    def test_invalid_owner_data_raises_error(self) -> None:
        with self.assertRaises(InvalidOperationError):
            Owner(full_name="", email="user@example.com", phone="+70000000000")

        with self.assertRaises(InvalidOperationError):
            Owner(full_name="User", email="invalid-email", phone="+70000000000")

    def test_savings_account_applies_monthly_interest(self) -> None:
        account = SavingsAccount(
            owner=self.owner,
            balance="1000.00",
            min_balance="100.00",
            monthly_interest_rate="5",
            currency="USD",
        )

        new_balance = account.apply_monthly_interest()

        self.assertEqual(new_balance, Decimal("1050.00"))
        self.assertIn("monthly_interest_rate", account.get_account_info())
        self.assertIn("SavingsAccount", str(account))

    def test_savings_account_respects_min_balance(self) -> None:
        account = SavingsAccount(
            owner=self.owner,
            balance="500.00",
            min_balance="200.00",
        )

        with self.assertRaises(InvalidOperationError):
            account.withdraw("350.00")

    def test_premium_account_allows_overdraft_and_charges_commission(self) -> None:
        account = PremiumAccount(
            owner=self.owner,
            balance="100.00",
            overdraft_limit="200.00",
            fixed_commission="10.00",
            single_withdrawal_limit="500.00",
            currency="EUR",
        )

        new_balance = account.withdraw("250.00")

        self.assertEqual(new_balance, Decimal("-160.00"))
        info = account.get_account_info()
        self.assertEqual(info["fixed_commission"], "10.00")
        self.assertIn("PremiumAccount", str(account))

    def test_premium_account_rejects_withdrawal_above_overdraft_limit(self) -> None:
        account = PremiumAccount(
            owner=self.owner,
            balance="100.00",
            overdraft_limit="50.00",
            fixed_commission="10.00",
            single_withdrawal_limit="500.00",
        )

        with self.assertRaises(InsufficientFundsError):
            account.withdraw("150.00")

    def test_investment_account_projects_growth(self) -> None:
        account = InvestmentAccount(
            owner=self.owner,
            balance="300.00",
            currency="USD",
            portfolio={"stocks": "1000.00", "bonds": "500.00", "etf": "500.00"},
        )

        projected_value = account.project_yearly_growth("10")

        self.assertEqual(projected_value, Decimal("2200.00"))
        info = account.get_account_info()
        self.assertEqual(info["portfolio_total"], "2000.00")
        self.assertIn("InvestmentAccount", str(account))

    def test_investment_account_can_withdraw_from_cash_and_portfolio(self) -> None:
        account = InvestmentAccount(
            owner=self.owner,
            balance="100.00",
            portfolio={"stocks": "150.00", "bonds": "50.00", "etf": "0.00"},
        )

        new_balance = account.withdraw("220.00")

        self.assertEqual(new_balance, Decimal("0.00"))
        self.assertEqual(account.portfolio["stocks"], Decimal("30.00"))
        self.assertEqual(account.portfolio["bonds"], Decimal("50.00"))


class BankManagerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.bank = Bank(name="My Bank")
        self.client = Client(
            full_name="Anna Smirnova",
            email="anna.smirnova@example.com",
            phone="+77005554433",
            age=28,
            pin_code="1234",
        )
        self.second_client = Client(
            full_name="Boris Ivanov",
            email="boris.ivanov@example.com",
            phone="+77009998877",
            age=35,
            pin_code="5678",
        )
        self.bank.add_client(self.client)
        self.bank.add_client(self.second_client)

    def test_client_requires_adult_age(self) -> None:
        with self.assertRaises(InvalidOperationError):
            Client(
                full_name="Young User",
                email="young.user@example.com",
                phone="+77001112233",
                age=17,
                pin_code="1234",
            )

    def test_open_account_creates_accounts_for_client(self) -> None:
        bank_account = self.bank.open_account(self.client.client_id, account_type="bank", balance="100.00")
        savings_account = self.bank.open_account(
            self.client.client_id,
            account_type="savings",
            balance="500.00",
            min_balance="50.00",
            monthly_interest_rate="3",
        )

        self.assertIsInstance(bank_account, BankAccount)
        self.assertIsInstance(savings_account, SavingsAccount)
        self.assertEqual(len(self.client.account_ids), 2)

    def test_open_account_regenerates_duplicate_account_id(self) -> None:
        first_account = self.bank.open_account(
            self.client.client_id,
            account_type="bank",
            balance="100.00",
            account_id="DUPL1234",
        )

        second_account = self.bank.open_account(
            self.second_client.client_id,
            account_type="bank",
            balance="200.00",
            account_id="DUPL1234",
        )

        self.assertEqual(first_account.account_id, "DUPL1234")
        self.assertNotEqual(second_account.account_id, "DUPL1234")
        self.assertIn(first_account.account_id, self.bank.accounts)
        self.assertIn(second_account.account_id, self.bank.accounts)
        self.assertIs(self.bank.accounts[first_account.account_id], first_account)
        self.assertIs(self.bank.accounts[second_account.account_id], second_account)
        self.assertEqual(self.bank.account_to_client[first_account.account_id], self.client.client_id)
        self.assertEqual(self.bank.account_to_client[second_account.account_id], self.second_client.client_id)

    def test_authenticate_client_locks_after_three_failed_attempts(self) -> None:
        with self.assertRaises(InvalidOperationError):
            self.bank.authenticate_client(self.client.client_id, "9999")
        with self.assertRaises(InvalidOperationError):
            self.bank.authenticate_client(self.client.client_id, "9999")
        with self.assertRaises(InvalidOperationError):
            self.bank.authenticate_client(self.client.client_id, "9999")

        self.assertTrue(self.client.is_locked)
        self.assertEqual(self.client.status, ClientStatus.BLOCKED)
        self.assertGreaterEqual(len(self.client.suspicious_activity), 1)

    def test_authenticate_client_succeeds_with_correct_pin(self) -> None:
        self.assertTrue(self.bank.authenticate_client(self.client.client_id, "1234"))
        self.assertEqual(self.client.failed_auth_attempts, 0)

    def test_freeze_unfreeze_and_close_account(self) -> None:
        account = self.bank.open_account(self.client.client_id, account_type="premium", balance="400.00")

        self.assertEqual(self.bank.freeze_account(account.account_id), AccountStatus.FROZEN)
        self.assertEqual(self.bank.unfreeze_account(account.account_id), AccountStatus.ACTIVE)
        self.assertEqual(self.bank.close_account(account.account_id), AccountStatus.CLOSED)

    def test_blocked_client_cannot_manage_account_state(self) -> None:
        account = self.bank.open_account(self.client.client_id, account_type="premium", balance="400.00")

        for _ in range(3):
            with self.assertRaises(InvalidOperationError):
                self.bank.authenticate_client(self.client.client_id, "9999")

        with self.assertRaises(InvalidOperationError):
            self.bank.freeze_account(account.account_id)

        self.assertEqual(account.status, AccountStatus.ACTIVE)

        account.freeze()

        with self.assertRaises(InvalidOperationError):
            self.bank.unfreeze_account(account.account_id)

        self.assertEqual(account.status, AccountStatus.FROZEN)

        with self.assertRaises(InvalidOperationError):
            self.bank.close_account(account.account_id)

        self.assertEqual(account.status, AccountStatus.FROZEN)

    def test_restricted_hours_block_operations_and_mark_suspicious(self) -> None:
        with patch.object(Bank, "_current_hour", return_value=1):
            with self.assertRaises(InvalidOperationError):
                self.bank.open_account(self.client.client_id, account_type="bank", balance="50.00")

        self.assertEqual(self.client.status, ClientStatus.SUSPICIOUS)
        self.assertIn("restricted hours operation: open_account", self.client.suspicious_activity)

    def test_search_accounts_filters_by_client_and_type(self) -> None:
        self.bank.open_account(self.client.client_id, account_type="bank", balance="100.00")
        self.bank.open_account(self.client.client_id, account_type="investment", portfolio={"stocks": "300.00"})
        self.bank.open_account(self.second_client.client_id, account_type="premium", balance="200.00")

        matched_accounts = self.bank.search_accounts(client_id=self.client.client_id, account_type="investment")

        self.assertEqual(len(matched_accounts), 1)
        self.assertIsInstance(matched_accounts[0], InvestmentAccount)

    def test_get_total_balance_and_clients_ranking(self) -> None:
        self.bank.open_account(self.client.client_id, account_type="bank", balance="100.00", currency="RUB")
        self.bank.open_account(
            self.client.client_id,
            account_type="investment",
            balance="50.00",
            currency="USD",
            portfolio={"stocks": "250.00", "bonds": "100.00", "etf": "50.00"},
        )
        self.bank.open_account(self.second_client.client_id, account_type="bank", balance="300.00", currency="USD")
        self.bank.open_account(self.second_client.client_id, account_type="bank", balance="200.00", currency="RUB")

        total_balance = self.bank.get_total_balance()
        ranking = self.bank.get_clients_ranking()

        self.assertEqual(
            total_balance,
            {
                "RUB": Decimal("300.00"),
                "USD": Decimal("750.00"),
                "EUR": Decimal("0.00"),
                "KZT": Decimal("0.00"),
                "CNY": Decimal("0.00"),
            },
        )
        self.assertEqual(ranking["RUB"][0]["full_name"], "Boris Ivanov")
        self.assertEqual(ranking["RUB"][0]["total_assets"], "200.00")
        self.assertEqual(ranking["RUB"][1]["total_assets"], "100.00")
        self.assertEqual(ranking["USD"][0]["full_name"], "Anna Smirnova")
        self.assertEqual(ranking["USD"][0]["total_assets"], "450.00")
        self.assertEqual(ranking["USD"][1]["total_assets"], "300.00")
        self.assertEqual(ranking["EUR"], [])
        self.assertEqual(ranking["KZT"], [])
        self.assertEqual(ranking["CNY"], [])


if __name__ == "__main__":
    unittest.main()
