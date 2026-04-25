
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import tempfile
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
    ReportBuilder,
    RiskLevel,
    SavingsAccount,
    Transaction,
    TransactionProcessor,
    TransactionQueue,
    TransactionStatus,
    TransactionType,
)
from src.main import build_demo_bank, generate_report_artifacts


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
        self.current_hour_patcher = patch.object(self.bank, "_current_hour", return_value=10)
        self.current_hour_patcher.start()
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

    def tearDown(self) -> None:
        self.current_hour_patcher.stop()

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
        failed_auth_events = self.bank.audit_log.filter_events(event_type="authentication_failed", client_id=self.client.client_id)
        blocked_events = self.bank.audit_log.filter_events(event_type="client_blocked", client_id=self.client.client_id)
        self.assertEqual(len(failed_auth_events), 3)
        self.assertEqual(len(blocked_events), 1)
        self.assertEqual(blocked_events[0].risk_level, RiskLevel.HIGH)

    def test_authenticate_client_succeeds_with_correct_pin(self) -> None:
        self.assertTrue(self.bank.authenticate_client(self.client.client_id, "1234"))
        self.assertEqual(self.client.failed_auth_attempts, 0)

    def test_freeze_unfreeze_and_close_account(self) -> None:
        account = self.bank.open_account(self.client.client_id, account_type="premium", balance="400.00")

        self.assertEqual(self.bank.freeze_account(account.account_id), AccountStatus.FROZEN)
        self.assertEqual(self.bank.unfreeze_account(account.account_id), AccountStatus.ACTIVE)
        self.assertEqual(self.bank.close_account(account.account_id), AccountStatus.CLOSED)

        account_events = [event for event in self.bank.audit_log.events if event.account_id == account.account_id]
        self.assertEqual(len([event for event in account_events if event.event_type == "account_opened"]), 1)
        self.assertEqual(len([event for event in account_events if event.event_type == "account_frozen"]), 1)
        self.assertEqual(len([event for event in account_events if event.event_type == "account_unfrozen"]), 1)
        self.assertEqual(len([event for event in account_events if event.event_type == "account_closed"]), 1)

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
        with patch.object(self.bank, "_current_hour", return_value=1):
            with self.assertRaises(InvalidOperationError):
                self.bank.open_account(self.client.client_id, account_type="bank", balance="50.00")

        self.assertEqual(self.client.status, ClientStatus.SUSPICIOUS)
        self.assertIn("restricted hours operation: open_account", self.client.suspicious_activity)
        suspicious_report = self.bank.get_audit_report_suspicious_operations(RiskLevel.MEDIUM)
        self.assertTrue(any(item["message"] == "restricted hours operation: open_account" for item in suspicious_report))

    def test_search_accounts_filters_by_client_and_type(self) -> None:
        self.bank.open_account(self.client.client_id, account_type="bank", balance="100.00")
        self.bank.open_account(self.client.client_id, account_type="investment", portfolio={"stocks": "300.00"})
        self.bank.open_account(self.second_client.client_id, account_type="premium", balance="200.00")

        matched_accounts = self.bank.search_accounts(client_id=self.client.client_id, account_type="investment")

        self.assertEqual(len(matched_accounts), 1)
        self.assertIsInstance(matched_accounts[0], InvestmentAccount)

    def test_get_total_balance_and_clients_ranking(self) -> None:
        with patch.object(self.bank, "_current_hour", return_value=10):
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


class TransactionQueueTestCase(unittest.TestCase):
    def test_transaction_uses_timezone_aware_utc_timestamps(self) -> None:
        transaction = Transaction(
            transaction_type=TransactionType.TRANSFER_EXTERNAL,
            amount="10.00",
            currency="RUB",
            sender_account_id="ACC10001",
            recipient_account_id="ACC20001",
        )

        self.assertIsNotNone(transaction.created_at.tzinfo)
        self.assertEqual(transaction.created_at.tzinfo, timezone.utc)
        self.assertIsNotNone(transaction.updated_at.tzinfo)
        self.assertEqual(transaction.updated_at.tzinfo, timezone.utc)

    def test_queue_orders_ready_transactions_by_priority_and_respects_schedule(self) -> None:
        queue = TransactionQueue()
        now = datetime.now(timezone.utc)
        low_priority_transaction = Transaction(
            transaction_type=TransactionType.TRANSFER_EXTERNAL,
            amount="10.00",
            currency="RUB",
            sender_account_id="ACC10001",
            recipient_account_id="ACC20001",
            priority=1,
        )
        high_priority_transaction = Transaction(
            transaction_type=TransactionType.TRANSFER_EXTERNAL,
            amount="20.00",
            currency="RUB",
            sender_account_id="ACC10002",
            recipient_account_id="ACC20002",
            priority=5,
        )
        delayed_transaction = Transaction(
            transaction_type=TransactionType.TRANSFER_EXTERNAL,
            amount="30.00",
            currency="RUB",
            sender_account_id="ACC10003",
            recipient_account_id="ACC20003",
            priority=10,
            scheduled_at=now + timedelta(hours=1),
        )

        queue.add_transaction(low_priority_transaction)
        queue.add_transaction(high_priority_transaction)
        queue.add_transaction(delayed_transaction)

        ready_transactions = queue.get_ready_transactions(now)

        self.assertEqual(
            [transaction.transaction_id for transaction in ready_transactions],
            [high_priority_transaction.transaction_id, low_priority_transaction.transaction_id],
        )
        self.assertEqual(delayed_transaction.status, TransactionStatus.SCHEDULED)

    def test_queue_cancels_pending_transaction(self) -> None:
        queue = TransactionQueue()
        transaction = Transaction(
            transaction_type=TransactionType.TRANSFER_INTERNAL,
            amount="15.00",
            currency="USD",
            sender_account_id="ACC30001",
            recipient_account_id="ACC30002",
        )

        queue.add_transaction(transaction)
        cancelled_transaction = queue.cancel_transaction(transaction.transaction_id, "user request")

        self.assertEqual(cancelled_transaction.status, TransactionStatus.CANCELLED)
        self.assertEqual(cancelled_transaction.failure_reason, "user request")


class TransactionProcessorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.audit_log_file_path = f"{self.temp_dir.name}/audit.jsonl"
        self.bank = Bank(name="Transaction Bank", audit_log_file_path=self.audit_log_file_path)
        self.first_client = Client(
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
        self.third_client = Client(
            full_name="Clara Kim",
            email="clara.kim@example.com",
            phone="+77007778899",
            age=31,
            pin_code="1111",
        )
        self.bank.add_client(self.first_client)
        self.bank.add_client(self.second_client)
        self.bank.add_client(self.third_client)
        with patch.object(self.bank, "_current_hour", return_value=10):
            self.first_rub = self.bank.open_account(
                self.first_client.client_id,
                account_type="bank",
                balance="1000.00",
                currency="RUB",
                account_id="A_RUB_01",
            )
            self.first_usd = self.bank.open_account(
                self.first_client.client_id,
                account_type="bank",
                balance="200.00",
                currency="USD",
                account_id="A_USD_01",
            )
            self.second_rub = self.bank.open_account(
                self.second_client.client_id,
                account_type="bank",
                balance="300.00",
                currency="RUB",
                account_id="B_RUB_01",
            )
            self.second_usd = self.bank.open_account(
                self.second_client.client_id,
                account_type="bank",
                balance="50.00",
                currency="USD",
                account_id="B_USD_01",
            )
            self.third_premium = self.bank.open_account(
                self.third_client.client_id,
                account_type="premium",
                balance="20.00",
                currency="RUB",
                overdraft_limit="200.00",
                fixed_commission="0.00",
                single_withdrawal_limit="500.00",
                account_id="C_PRM_01",
            )
        self.processor = TransactionProcessor(
            self.bank,
            exchange_rates={
                ("RUB", "USD"): "0.01",
                ("USD", "RUB"): "100.00",
                ("EUR", "USD"): "1.10",
            },
            external_transfer_fee="10.00",
            max_retries=3,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_audit_log_persists_events_in_memory_and_file(self) -> None:
        transaction = Transaction(
            transaction_type=TransactionType.TRANSFER_EXTERNAL,
            amount="100.00",
            currency="RUB",
            sender_account_id=self.first_rub.account_id,
            recipient_account_id=self.second_usd.account_id,
        )

        self.processor.process_transaction(transaction)

        risk_events = self.bank.audit_log.filter_events(
            event_type="risk_assessment",
            transaction_id=transaction.transaction_id,
        )
        completed_events = self.bank.audit_log.filter_events(
            event_type="transaction_completed",
            transaction_id=transaction.transaction_id,
        )
        self.assertEqual(len(risk_events), 1)
        self.assertEqual(len(completed_events), 1)
        with open(self.audit_log_file_path, encoding="utf-8") as file:
            lines = [json.loads(line) for line in file if line.strip()]
        self.assertGreaterEqual(len(lines), 1)
        self.assertTrue(any(line["transaction_id"] == transaction.transaction_id for line in lines))

    def test_audit_report_and_risk_profile_include_blocked_operation(self) -> None:
        self.bank.risk_analyzer.large_amount_threshold = Decimal("5000.00")
        blocked_time = datetime(2026, 4, 16, 1, 10, tzinfo=timezone.utc)
        transaction = Transaction(
            transaction_type=TransactionType.TRANSFER_EXTERNAL,
            amount="6000.00",
            currency="RUB",
            sender_account_id=self.first_rub.account_id,
            recipient_account_id=self.second_rub.account_id,
        )

        processed_transaction = self.processor.process_transaction(transaction, now=blocked_time)

        self.assertEqual(processed_transaction.status, TransactionStatus.FAILED)
        suspicious_report = self.bank.get_audit_report_suspicious_operations(RiskLevel.MEDIUM)
        self.assertTrue(any(event["transaction_id"] == transaction.transaction_id for event in suspicious_report))
        risk_profile = self.bank.get_client_risk_profile(self.first_client.client_id)
        self.assertEqual(risk_profile["highest_risk"], RiskLevel.HIGH.value)
        self.assertGreaterEqual(risk_profile["blocked_operations_count"], 1)
        self.assertIn(self.second_rub.account_id, risk_profile["recent_risky_recipients"])

    def test_audit_error_statistics_count_blocked_and_failed_operations(self) -> None:
        self.bank.risk_analyzer.large_amount_threshold = Decimal("2000.00")
        blocked_time = datetime(2026, 4, 16, 2, 0, tzinfo=timezone.utc)
        blocked_transaction = Transaction(
            transaction_type=TransactionType.TRANSFER_EXTERNAL,
            amount="2500.00",
            currency="RUB",
            sender_account_id=self.first_rub.account_id,
            recipient_account_id=self.second_rub.account_id,
        )
        failed_transaction = Transaction(
            transaction_type=TransactionType.TRANSFER_EXTERNAL,
            amount="20.00",
            currency="RUB",
            sender_account_id=self.first_rub.account_id,
            recipient_account_id=self.second_rub.account_id,
        )
        self.second_rub.freeze()

        self.processor.process_transaction(blocked_transaction, now=blocked_time)
        self.processor.process_transaction(failed_transaction, now=datetime(2026, 4, 16, 8, 0, tzinfo=timezone.utc))

        error_statistics = self.bank.get_audit_error_statistics()
        self.assertEqual(error_statistics["operation blocked by risk analyzer"], 1)
        self.assertEqual(error_statistics["recipient account is frozen"], 1)

    def test_risk_analyzer_blocks_high_risk_transaction(self) -> None:
        self.bank.risk_analyzer.large_amount_threshold = Decimal("5000.00")
        high_risk_time = datetime(2026, 4, 16, 1, 0, tzinfo=timezone.utc)
        transaction = Transaction(
            transaction_type=TransactionType.TRANSFER_EXTERNAL,
            amount="7000.00",
            currency="RUB",
            sender_account_id=self.first_rub.account_id,
            recipient_account_id=self.second_rub.account_id,
        )

        processed_transaction = self.processor.process_transaction(transaction, now=high_risk_time)

        self.assertEqual(processed_transaction.status, TransactionStatus.FAILED)
        self.assertEqual(processed_transaction.failure_reason, "operation blocked by risk analyzer")
        blocked_events = self.bank.audit_log.filter_events(
            event_type="transaction_blocked",
            transaction_id=transaction.transaction_id,
        )
        self.assertEqual(len(blocked_events), 1)
        self.assertEqual(blocked_events[0].risk_level, RiskLevel.HIGH)
        self.assertEqual(self.first_client.status, ClientStatus.SUSPICIOUS)

    def test_processor_handles_external_fee_and_currency_conversion(self) -> None:
        transaction = Transaction(
            transaction_type=TransactionType.TRANSFER_EXTERNAL,
            amount="100.00",
            currency="RUB",
            sender_account_id=self.first_rub.account_id,
            recipient_account_id=self.second_usd.account_id,
        )

        processed_transaction = self.processor.process_transaction(transaction)

        self.assertEqual(processed_transaction.status, TransactionStatus.COMPLETED)
        self.assertEqual(processed_transaction.fee, Decimal("10.00"))
        self.assertEqual(self.first_rub.balance, Decimal("890.00"))
        self.assertEqual(self.second_usd.balance, Decimal("51.00"))

    def test_processor_rejects_frozen_account_transactions(self) -> None:
        self.second_rub.freeze()
        transaction = Transaction(
            transaction_type=TransactionType.TRANSFER_EXTERNAL,
            amount="20.00",
            currency="RUB",
            sender_account_id=self.first_rub.account_id,
            recipient_account_id=self.second_rub.account_id,
        )

        processed_transaction = self.processor.process_transaction(transaction)

        self.assertEqual(processed_transaction.status, TransactionStatus.FAILED)
        self.assertEqual(processed_transaction.failure_reason, "recipient account is frozen")
        self.assertEqual(self.first_rub.balance, Decimal("1000.00"))
        self.assertEqual(self.second_rub.balance, Decimal("300.00"))

    def test_processor_allows_premium_overdraft_for_external_transfer(self) -> None:
        transaction = Transaction(
            transaction_type=TransactionType.TRANSFER_EXTERNAL,
            amount="100.00",
            currency="RUB",
            sender_account_id=self.third_premium.account_id,
            recipient_account_id=self.second_rub.account_id,
        )

        processed_transaction = self.processor.process_transaction(transaction)

        self.assertEqual(processed_transaction.status, TransactionStatus.COMPLETED)
        self.assertEqual(self.third_premium.balance, Decimal("-90.00"))
        self.assertEqual(self.second_rub.balance, Decimal("400.00"))

    def test_processor_marks_retrying_for_temporary_errors(self) -> None:
        transaction = Transaction(
            transaction_type=TransactionType.TRANSFER_INTERNAL,
            amount="10.00",
            currency="RUB",
            sender_account_id=self.first_rub.account_id,
            recipient_account_id=self.second_rub.account_id,
        )

        with patch.object(self.processor, "_execute_transaction", side_effect=RuntimeError("temporary processor error")):
            processed_transaction = self.processor.process_transaction(transaction)

        self.assertEqual(processed_transaction.status, TransactionStatus.RETRYING)
        self.assertEqual(processed_transaction.retry_count, 1)
        self.assertIn("temporary processor error", processed_transaction.error_log)

    def test_process_until_idle_does_not_retry_same_transaction_in_one_call(self) -> None:
        queue = TransactionQueue()
        transaction = Transaction(
            transaction_type=TransactionType.TRANSFER_INTERNAL,
            amount="10.00",
            currency="RUB",
            sender_account_id=self.first_rub.account_id,
            recipient_account_id=self.second_rub.account_id,
        )
        queue.add_transaction(transaction)

        with patch.object(self.processor, "_execute_transaction", side_effect=RuntimeError("temporary processor error")):
            processed_transactions = self.processor.process_until_idle(queue)

        self.assertEqual(len(processed_transactions), 1)
        self.assertIs(processed_transactions[0], transaction)
        self.assertEqual(transaction.status, TransactionStatus.RETRYING)
        self.assertEqual(transaction.retry_count, 1)

    def test_processor_marks_unknown_sender_account_as_failed_and_audited(self) -> None:
        transaction = Transaction(
            transaction_type=TransactionType.TRANSFER_EXTERNAL,
            amount="10.00",
            currency="RUB",
            sender_account_id="UNKNOWN1",
            recipient_account_id=self.second_rub.account_id,
        )

        processed_transaction = self.processor.process_transaction(transaction)

        self.assertEqual(processed_transaction.status, TransactionStatus.FAILED)
        self.assertEqual(processed_transaction.failure_reason, "account not linked to client")
        failed_events = self.bank.audit_log.filter_events(
            event_type="transaction_failed",
            transaction_id=transaction.transaction_id,
        )
        self.assertEqual(len(failed_events), 1)
        self.assertIsNone(failed_events[0].client_id)

    def test_risk_analyzer_normalizes_large_amounts_across_currencies(self) -> None:
        with patch.object(self.bank, "_current_hour", return_value=10):
            first_kzt = self.bank.open_account(
                self.first_client.client_id,
                account_type="bank",
                balance="10000.00",
                currency="KZT",
                account_id="A_KZT_01",
            )
            second_kzt = self.bank.open_account(
                self.second_client.client_id,
                account_type="bank",
                balance="5000.00",
                currency="KZT",
                account_id="B_KZT_01",
            )
        self.bank.risk_analyzer.large_amount_threshold = Decimal("1000.00")
        midday = datetime(2026, 4, 16, 12, 0, tzinfo=timezone.utc)
        kzt_transaction = Transaction(
            transaction_type=TransactionType.TRANSFER_EXTERNAL,
            amount="1500.00",
            currency="KZT",
            sender_account_id=first_kzt.account_id,
            recipient_account_id=second_kzt.account_id,
        )
        usd_transaction = Transaction(
            transaction_type=TransactionType.TRANSFER_EXTERNAL,
            amount="1500.00",
            currency="USD",
            sender_account_id=self.first_usd.account_id,
            recipient_account_id=self.second_usd.account_id,
        )

        kzt_risk = self.bank.risk_analyzer.analyze_transaction(kzt_transaction, self.bank, midday)
        usd_risk = self.bank.risk_analyzer.analyze_transaction(usd_transaction, self.bank, midday)

        self.assertNotIn("large amount", kzt_risk.reasons)
        self.assertIn("large amount", usd_risk.reasons)

    def test_current_hour_uses_utc_for_aware_datetimes(self) -> None:
        self.assertEqual(self.bank._current_hour(datetime(2026, 4, 16, 0, 30, tzinfo=timezone(timedelta(hours=5)))), 19)

    def test_processor_blocks_transactions_during_restricted_hours_and_marks_sender_suspicious(self) -> None:
        restricted_time = datetime(2026, 4, 16, 1, 30, tzinfo=timezone.utc)
        transaction = Transaction(
            transaction_type=TransactionType.TRANSFER_EXTERNAL,
            amount="100.00",
            currency="RUB",
            sender_account_id=self.first_rub.account_id,
            recipient_account_id=self.second_rub.account_id,
        )

        processed_transaction = self.processor.process_transaction(transaction, now=restricted_time)

        self.assertEqual(processed_transaction.status, TransactionStatus.FAILED)
        self.assertEqual(processed_transaction.failure_reason, "operations are not allowed from 00:00 to 05:00")
        self.assertEqual(self.first_rub.balance, Decimal("1000.00"))
        self.assertEqual(self.second_rub.balance, Decimal("300.00"))
        self.assertEqual(self.first_client.status, ClientStatus.SUSPICIOUS)
        self.assertIn(
            "restricted hours operation: transaction_transfer_external",
            self.first_client.suspicious_activity,
        )

    def test_scheduled_transaction_fails_if_execution_time_is_restricted(self) -> None:
        queue = TransactionQueue()
        scheduled_time = datetime(2026, 4, 17, 1, 0, tzinfo=timezone.utc)
        transaction = Transaction(
            transaction_type=TransactionType.TRANSFER_EXTERNAL,
            amount="50.00",
            currency="RUB",
            sender_account_id=self.first_rub.account_id,
            recipient_account_id=self.second_rub.account_id,
            scheduled_at=scheduled_time,
        )

        queue.add_transaction(transaction)
        processed_transactions = self.processor.process_until_idle(queue, now=scheduled_time)

        self.assertEqual(len(processed_transactions), 1)
        self.assertEqual(transaction.status, TransactionStatus.FAILED)
        self.assertEqual(transaction.failure_reason, "operations are not allowed from 00:00 to 05:00")
        self.assertEqual(self.first_rub.balance, Decimal("1000.00"))
        self.assertEqual(self.second_rub.balance, Decimal("300.00"))
        self.assertIn(
            "restricted hours operation: transaction_transfer_external",
            self.first_client.suspicious_activity,
        )

    def test_process_queue_executes_ten_transactions(self) -> None:
        queue = TransactionQueue()
        now = datetime.now(timezone.utc)
        transactions = [
            Transaction(
                transaction_type=TransactionType.TRANSFER_INTERNAL,
                amount="50.00",
                currency="RUB",
                sender_account_id=self.first_rub.account_id,
                recipient_account_id=self.second_rub.account_id,
                priority=8,
            ),
            Transaction(
                transaction_type=TransactionType.TRANSFER_EXTERNAL,
                amount="200.00",
                currency="RUB",
                sender_account_id=self.first_rub.account_id,
                recipient_account_id=self.second_rub.account_id,
                priority=7,
            ),
            Transaction(
                transaction_type=TransactionType.EXCHANGE,
                amount="100.00",
                currency="RUB",
                sender_account_id=self.first_rub.account_id,
                recipient_account_id=self.first_usd.account_id,
                priority=9,
            ),
            Transaction(
                transaction_type=TransactionType.TRANSFER_EXTERNAL,
                amount="30.00",
                currency="USD",
                sender_account_id=self.first_usd.account_id,
                recipient_account_id=self.second_usd.account_id,
                priority=6,
            ),
            Transaction(
                transaction_type=TransactionType.TRANSFER_EXTERNAL,
                amount="40.00",
                currency="RUB",
                sender_account_id=self.first_rub.account_id,
                recipient_account_id=self.third_premium.account_id,
                priority=5,
            ),
            Transaction(
                transaction_type=TransactionType.TRANSFER_INTERNAL,
                amount="60.00",
                currency="USD",
                sender_account_id=self.first_usd.account_id,
                recipient_account_id=self.first_rub.account_id,
                priority=4,
            ),
            Transaction(
                transaction_type=TransactionType.TRANSFER_EXTERNAL,
                amount="150.00",
                currency="RUB",
                sender_account_id=self.third_premium.account_id,
                recipient_account_id=self.second_rub.account_id,
                priority=3,
            ),
            Transaction(
                transaction_type=TransactionType.TRANSFER_EXTERNAL,
                amount="5000.00",
                currency="RUB",
                sender_account_id=self.first_rub.account_id,
                recipient_account_id=self.second_rub.account_id,
                priority=2,
            ),
            Transaction(
                transaction_type=TransactionType.TRANSFER_EXTERNAL,
                amount="25.00",
                currency="RUB",
                sender_account_id=self.first_rub.account_id,
                recipient_account_id=self.second_rub.account_id,
                priority=10,
                scheduled_at=now + timedelta(minutes=30),
            ),
            Transaction(
                transaction_type=TransactionType.TRANSFER_EXTERNAL,
                amount="15.00",
                currency="RUB",
                sender_account_id=self.first_rub.account_id,
                recipient_account_id=self.second_rub.account_id,
                priority=1,
            ),
        ]
        cancelled_transaction = transactions[9]

        for transaction in transactions:
            queue.add_transaction(transaction)

        queue.cancel_transaction(cancelled_transaction.transaction_id, "user cancelled before execution")
        self.second_usd.freeze()

        processed_before_schedule = self.processor.process_until_idle(queue, now)

        self.assertEqual(len(processed_before_schedule), 8)
        self.assertEqual(transactions[0].status, TransactionStatus.FAILED)
        self.assertEqual(transactions[1].status, TransactionStatus.COMPLETED)
        self.assertEqual(transactions[2].status, TransactionStatus.COMPLETED)
        self.assertEqual(transactions[3].status, TransactionStatus.FAILED)
        self.assertEqual(transactions[4].status, TransactionStatus.COMPLETED)
        self.assertEqual(transactions[5].status, TransactionStatus.FAILED)
        self.assertEqual(transactions[6].status, TransactionStatus.COMPLETED)
        self.assertEqual(transactions[7].status, TransactionStatus.FAILED)
        self.assertEqual(transactions[8].status, TransactionStatus.SCHEDULED)
        self.assertEqual(transactions[9].status, TransactionStatus.CANCELLED)
        self.assertEqual(transactions[0].failure_reason, "internal transfer requires accounts of the same client")
        self.assertEqual(transactions[3].failure_reason, "operation blocked by risk analyzer")
        self.assertEqual(transactions[5].failure_reason, "operation blocked by risk analyzer")
        self.assertEqual(transactions[7].failure_reason, "insufficient funds")

        self.second_usd.activate()
        processed_after_schedule = self.processor.process_until_idle(queue, now + timedelta(hours=1))

        self.assertEqual(len(processed_after_schedule), 1)
        self.assertEqual(transactions[8].status, TransactionStatus.COMPLETED)
        self.assertEqual(self.first_rub.balance, Decimal("605.00"))
        self.assertEqual(self.first_usd.balance, Decimal("201.00"))
        self.assertEqual(self.second_rub.balance, Decimal("675.00"))
        self.assertEqual(self.second_usd.balance, Decimal("50.00"))
        self.assertEqual(self.third_premium.balance, Decimal("-100.00"))

    def test_cancel_completed_transaction_is_not_allowed(self) -> None:
        queue = TransactionQueue()
        transaction = Transaction(
            transaction_type=TransactionType.TRANSFER_EXTERNAL,
            amount="20.00",
            currency="RUB",
            sender_account_id=self.first_rub.account_id,
            recipient_account_id=self.second_rub.account_id,
        )

        queue.add_transaction(transaction)
        self.processor.process_transaction(transaction)

        with self.assertRaises(InvalidOperationError):
            queue.cancel_transaction(transaction.transaction_id)


class ReportBuilderTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.audit_log_file_path = f"{self.temp_dir.name}/audit.jsonl"
        self.bank = Bank(name="Reporting Bank", audit_log_file_path=self.audit_log_file_path)
        self.first_client = Client(
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
        self.bank.add_client(self.first_client)
        self.bank.add_client(self.second_client)
        with patch.object(self.bank, "_current_hour", return_value=10):
            self.first_rub = self.bank.open_account(
                self.first_client.client_id,
                account_type="bank",
                balance="1000.00",
                currency="RUB",
                account_id="RB_RUB_01",
            )
            self.first_usd = self.bank.open_account(
                self.first_client.client_id,
                account_type="savings",
                balance="200.00",
                currency="USD",
                account_id="RB_USD_01",
                min_balance="50.00",
                monthly_interest_rate="0.01",
            )
            self.second_rub = self.bank.open_account(
                self.second_client.client_id,
                account_type="bank",
                balance="300.00",
                currency="RUB",
                account_id="RB_RUB_02",
            )
        self.processor = TransactionProcessor(
            self.bank,
            exchange_rates={
                ("RUB", "USD"): "0.01",
                ("USD", "RUB"): "100.00",
            },
        )
        self.completed_transaction = Transaction(
            transaction_type=TransactionType.TRANSFER_EXTERNAL,
            amount="100.00",
            currency="RUB",
            sender_account_id=self.first_rub.account_id,
            recipient_account_id=self.second_rub.account_id,
        )
        self.failed_transaction = Transaction(
            transaction_type=TransactionType.TRANSFER_EXTERNAL,
            amount="20.00",
            currency="RUB",
            sender_account_id=self.first_rub.account_id,
            recipient_account_id=self.second_rub.account_id,
        )
        self.blocked_transaction = Transaction(
            transaction_type=TransactionType.TRANSFER_EXTERNAL,
            amount="3000.00",
            currency="RUB",
            sender_account_id=self.first_rub.account_id,
            recipient_account_id=self.second_rub.account_id,
        )
        self.processor.process_transaction(self.completed_transaction, now=datetime(2026, 4, 18, 8, 0, tzinfo=timezone.utc))
        self.second_rub.freeze()
        self.processor.process_transaction(self.failed_transaction, now=datetime(2026, 4, 18, 9, 0, tzinfo=timezone.utc))
        self.bank.risk_analyzer.large_amount_threshold = Decimal("2000.00")
        self.processor.process_transaction(self.blocked_transaction, now=datetime(2026, 4, 18, 1, 0, tzinfo=timezone.utc))
        self.report_builder = ReportBuilder(self.bank)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_build_client_report_contains_accounts_and_risk_data(self) -> None:
        report = self.report_builder.build_client_report(self.first_client.client_id)

        self.assertEqual(report["report_type"], "client")
        self.assertEqual(report["client"]["client_id"], self.first_client.client_id)
        self.assertEqual(len(report["accounts"]), 2)
        self.assertIn("RUB", report["total_assets_by_currency"])
        self.assertGreaterEqual(len(report["transactions"]), 2)
        self.assertIn("highest_risk", report["risk_profile"])

    def test_build_client_report_includes_incoming_transfer_audit_for_recipient(self) -> None:
        report = self.report_builder.build_client_report(self.second_client.client_id)

        self.assertTrue(any(event["event_type"] == "transaction_completed" for event in report["audit_events"]))
        self.assertTrue(
            any(
                event["metadata"].get("recipient_client_id") == self.second_client.client_id
                for event in report["audit_events"]
                if isinstance(event.get("metadata"), dict)
            )
        )

    def test_build_bank_report_contains_totals_rankings_and_statistics(self) -> None:
        report = self.report_builder.build_bank_report()

        self.assertEqual(report["report_type"], "bank")
        self.assertEqual(report["clients_count"], 2)
        self.assertEqual(report["accounts_count"], 3)
        self.assertIn("RUB", report["total_balance"])
        self.assertIn("RUB", report["client_rankings"])
        self.assertIn("by_status", report["transaction_statistics"])
        self.assertIn("recipient account is frozen", report["audit_error_statistics"])

    def test_build_risk_report_contains_suspicious_and_blocked_operations(self) -> None:
        report = self.report_builder.build_risk_report()

        self.assertEqual(report["report_type"], "risk")
        self.assertGreaterEqual(report["blocked_operations_count"], 1)
        self.assertGreaterEqual(len(report["suspicious_operations"]), 1)
        self.assertIn("high", report["risk_level_distribution"])
        self.assertGreaterEqual(len(report["top_risky_clients"]), 1)

    def test_export_to_json_and_csv_writes_files(self) -> None:
        report = self.report_builder.build_bank_report()
        json_path = Path(self.temp_dir.name) / "bank_report.json"
        csv_path = Path(self.temp_dir.name) / "bank_report.csv"

        exported_json_path = self.report_builder.export_to_json(report, str(json_path))
        exported_csv_path = self.report_builder.export_to_csv(report, str(csv_path))

        self.assertTrue(Path(exported_json_path).exists())
        self.assertTrue(Path(exported_csv_path).exists())
        with json_path.open(encoding="utf-8") as file:
            exported_json = json.load(file)
        self.assertEqual(exported_json["report_type"], "bank")
        with csv_path.open(encoding="utf-8") as file:
            csv_content = file.read()
        self.assertIn("field,value", csv_content)
        self.assertIn("bank_name,Reporting Bank", csv_content)

    def test_render_text_returns_json_like_text(self) -> None:
        report = self.report_builder.build_risk_report()

        rendered_text = self.report_builder.render_text(report)

        self.assertIn('"report_type": "risk"', rendered_text)
        self.assertIn('"blocked_operations_count"', rendered_text)

    def test_save_charts_writes_real_png_files(self) -> None:
        charts_dir = Path(self.temp_dir.name) / "charts"

        saved_paths = self.report_builder.save_charts(str(charts_dir), client_id=self.first_client.client_id)

        self.assertEqual(len(saved_paths), 5)
        self.assertTrue(all(Path(path).exists() for path in saved_paths))
        self.assertTrue(all(Path(path).suffix == ".png" for path in saved_paths))
        self.assertTrue(all(Path(path).stat().st_size > 0 for path in saved_paths))
        self.assertTrue(any(path.endswith("bank_balance_movement.png") for path in saved_paths))
        self.assertTrue(any(path.endswith(f"client_{self.first_client.client_id}_balance_movement.png") for path in saved_paths))


class MainDemoIntegrationTestCase(unittest.TestCase):
    def test_generate_report_artifacts_writes_report_files_and_charts(self) -> None:
        bank, _processor, _queue, clients, _accounts, _phase_times, audit_log_path = build_demo_bank()

        artifacts = generate_report_artifacts(bank, clients, audit_log_path, client_keys=("alice", "boris"))

        output_dir = Path(str(artifacts["output_dir"]))
        self.assertTrue(output_dir.exists())
        self.assertTrue(Path(dict(artifacts["bank_report"])["json"]).exists())
        self.assertTrue(Path(dict(artifacts["bank_report"])["csv"]).exists())
        self.assertTrue(Path(dict(artifacts["risk_report"])["json"]).exists())
        self.assertTrue(Path(dict(artifacts["risk_report"])["csv"]).exists())
        client_reports = dict(artifacts["client_reports"])
        self.assertEqual(set(client_reports.keys()), {"alice", "boris"})
        self.assertTrue(all(Path(file_set["json"]).exists() for file_set in client_reports.values()))
        self.assertTrue(all(Path(file_set["csv"]).exists() for file_set in client_reports.values()))
        chart_paths = list(artifacts["charts"])
        self.assertEqual(len(chart_paths), 5)
        self.assertTrue(all(Path(path).exists() for path in chart_paths))
        self.assertIn('"report_type": "risk"', str(artifacts["risk_preview"]))


if __name__ == "__main__":
    unittest.main()
