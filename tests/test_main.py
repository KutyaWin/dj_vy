
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import csv
import json
from pathlib import Path
import sqlite3
import socket
import tempfile
import time
import unittest
from unittest.mock import patch

from aiohttp import web

from src.models import (
    AccountClosedError,
    AccountFrozenError,
    AccountStatus,
    AsyncCrawler,
    BankAccount,
    Bank,
    Client,
    ClientStatus,
    CrawlerQueue,
    Currency,
    InvestmentAccount,
    JSONStorage,
    HTMLParser,
    InsufficientFundsError,
    InvalidOperationError,
    NetworkError,
    Owner,
    ParseError,
    PermanentError,
    PremiumAccount,
    RateLimiter,
    ReportBuilder,
    RiskLevel,
    RetryStrategy,
    RobotsParser,
    SavingsAccount,
    SQLiteStorage,
    SemaphoreManager,
    TransientError,
    Transaction,
    TransactionProcessor,
    TransactionQueue,
    TransactionStatus,
    TransactionType,
    CSVStorage,
)
from src.main import build_demo_bank, fetch_urls_sequentially, generate_report_artifacts, run_async_crawler_demo, run_html_parser_demo, run_polite_crawl_demo, run_retry_demo, run_site_crawl_demo, run_storage_demo


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
            monthly_interest_rate="0.05",
            currency="USD",
        )

        new_balance = account.apply_monthly_interest()

        self.assertEqual(new_balance, Decimal("1050.00"))
        self.assertIn("monthly_interest_rate", account.get_account_info())
        self.assertIn("SavingsAccount", str(account))

    def test_savings_account_rate_uses_fractional_contract(self) -> None:
        account = SavingsAccount(
            owner=self.owner,
            balance="100.00",
            min_balance="0.00",
            monthly_interest_rate="0.01",
            currency="USD",
        )

        self.assertEqual(account.monthly_interest_rate, Decimal("0.0100"))
        self.assertEqual(account.get_account_info()["monthly_interest_rate"], "1.00%")

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

        projected_value = account.project_yearly_growth("0.10")

        self.assertEqual(projected_value, Decimal("2200.00"))
        info = account.get_account_info()
        self.assertEqual(info["portfolio_total"], "2000.00")
        self.assertIn("InvestmentAccount", str(account))

    def test_investment_account_growth_rate_uses_fractional_contract(self) -> None:
        account = InvestmentAccount(
            owner=self.owner,
            balance="0.00",
            currency="USD",
            portfolio={"stocks": "1000.00", "bonds": "0.00", "etf": "0.00"},
        )

        projected_value = account.project_yearly_growth("0.10")

        self.assertEqual(projected_value, Decimal("1100.00"))

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
            monthly_interest_rate="0.03",
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

    def test_processor_uses_total_fee_for_premium_sender_and_logs_fee_breakdown(self) -> None:
        with patch.object(self.bank, "_current_hour", return_value=10):
            premium_sender = self.bank.open_account(
                self.third_client.client_id,
                account_type="premium",
                balance="200.00",
                currency="RUB",
                overdraft_limit="0.00",
                fixed_commission="5.00",
                single_withdrawal_limit="500.00",
                account_id="C_PRM_02",
            )
        transaction = Transaction(
            transaction_type=TransactionType.TRANSFER_EXTERNAL,
            amount="100.00",
            currency="RUB",
            sender_account_id=premium_sender.account_id,
            recipient_account_id=self.second_rub.account_id,
        )

        processed_transaction = self.processor.process_transaction(transaction, now=datetime(2026, 4, 16, 12, 0, tzinfo=timezone.utc))

        self.assertEqual(processed_transaction.status, TransactionStatus.COMPLETED)
        self.assertEqual(processed_transaction.fee, Decimal("15.00"))
        self.assertEqual(premium_sender.balance, Decimal("85.00"))
        completed_events = self.bank.audit_log.filter_events(
            event_type="transaction_completed",
            transaction_id=transaction.transaction_id,
        )
        self.assertEqual(len(completed_events), 1)
        self.assertEqual(completed_events[0].metadata["processor_fee"], "10.00")
        self.assertEqual(completed_events[0].metadata["account_fee"], "5.00")
        self.assertEqual(completed_events[0].metadata["total_fee"], "15.00")
        self.assertEqual(completed_events[0].metadata["sender_total_charge"], "115.00")

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

    def test_client_balance_timeline_uses_actual_premium_sender_charge(self) -> None:
        with patch.object(self.bank, "_current_hour", return_value=10):
            premium_sender = self.bank.open_account(
                self.first_client.client_id,
                account_type="premium",
                balance="200.00",
                currency="RUB",
                overdraft_limit="0.00",
                fixed_commission="5.00",
                single_withdrawal_limit="500.00",
                account_id="RB_PRM_03",
            )
        premium_transaction = Transaction(
            transaction_type=TransactionType.TRANSFER_EXTERNAL,
            amount="100.00",
            currency="RUB",
            sender_account_id=premium_sender.account_id,
            recipient_account_id=self.second_rub.account_id,
        )

        self.second_rub.activate()
        self.processor.process_transaction(premium_transaction, now=datetime(2026, 4, 18, 10, 0, tzinfo=timezone.utc))
        timeline = self.report_builder._build_client_balance_timeline(self.first_client.client_id)
        current_index = timeline["labels"].index("current")

        self.assertEqual(timeline["series"]["RUB"][current_index], 975.0)
        self.assertEqual(timeline["series"]["RUB"][current_index - 1] - timeline["series"]["RUB"][current_index], 115.0)
        completed_events = self.bank.audit_log.filter_events(
            event_type="transaction_completed",
            transaction_id=premium_transaction.transaction_id,
        )
        self.assertEqual(len(completed_events), 1)
        self.assertEqual(completed_events[0].metadata["sender_total_charge"], "115.00")

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


class HTMLParserTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_parse_html_extracts_expected_fields(self) -> None:
        parser = HTMLParser()
        html = """
        <html>
            <head>
                <title>Example Domain</title>
                <meta name="description" content="Example description">
                <meta name="keywords" content="example, domain">
            </head>
            <body>
                <h1>Main title</h1>
                <h2>Section title</h2>
                <p>Hello <strong>world</strong></p>
                <a href="/about">About</a>
                <a href="https://external.example.org/page">External</a>
                <img src="/static/logo.png" alt="Logo">
                <table>
                    <tr><th>Name</th><th>Value</th></tr>
                    <tr><td>Alpha</td><td>1</td></tr>
                </table>
                <ul><li>One</li><li>Two</li></ul>
            </body>
        </html>
        """

        result = await parser.parse_html(html, "https://example.com/start")

        self.assertEqual(result["title"], "Example Domain")
        self.assertEqual(dict(result["metadata"])["description"], "Example description")
        self.assertEqual(dict(result["metadata"])["keywords"], "example, domain")
        self.assertIn("Hello world", str(result["text"]))
        self.assertEqual(list(result["links"]), ["https://example.com/about", "https://external.example.org/page"])
        self.assertEqual(list(result["images"]), [{"src": "https://example.com/static/logo.png", "alt": "Logo"}])
        self.assertEqual(list(result["headings"]), [{"tag": "h1", "text": "Main title"}, {"tag": "h2", "text": "Section title"}])
        self.assertEqual(list(result["tables"]), [[ ["Name", "Value"], ["Alpha", "1"] ]])
        self.assertEqual(list(result["lists"]), [{"type": "ul", "items": ["One", "Two"]}])

    async def test_parse_html_handles_broken_html_without_raising(self) -> None:
        parser = HTMLParser()
        html = "<html><head><title>Broken<title><body><h1>Oops<p><a href='/x'>go"

        result = await parser.parse_html(html, "https://example.com")

        self.assertEqual(result["url"], "https://example.com")
        self.assertIn("Oops", str(result["text"]))
        self.assertIn("https://example.com/x", list(result["links"]))

    def test_extract_links_converts_relative_urls_and_filters_invalid(self) -> None:
        parser = HTMLParser()
        soup = BeautifulSoup(
            """
            <html><body>
                <a href="/relative">Relative</a>
                <a href="contact">Contact</a>
                <a href="#fragment">Fragment</a>
                <a href="mailto:test@example.com">Mail</a>
                <a href="javascript:void(0)">JS</a>
                <a href="https://example.com/absolute">Absolute</a>
            </body></html>
            """,
            "html.parser",
        )

        links = parser.extract_links(soup, "https://example.com/base/index.html")

        self.assertEqual(
            links,
            [
                "https://example.com/relative",
                "https://example.com/base/contact",
                "https://example.com/absolute",
            ],
        )

    def test_extract_links_can_limit_to_same_domain(self) -> None:
        parser = HTMLParser(same_domain_only=True)
        soup = BeautifulSoup(
            '<a href="/docs">Docs</a><a href="https://other.example.org/page">External</a>',
            "html.parser",
        )

        links = parser.extract_links(soup, "https://example.com/start")

        self.assertEqual(links, ["https://example.com/docs"])

    def test_extract_text_with_selector_returns_scoped_text(self) -> None:
        parser = HTMLParser()
        soup = BeautifulSoup('<body><main><p>Target text</p></main><footer>Ignore me</footer></body>', "html.parser")

        text = parser.extract_text(soup, "main")

        self.assertEqual(text, "Target text")


class CrawlerQueueTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_queue_returns_urls_by_priority(self) -> None:
        queue = CrawlerQueue()

        await queue.add_url("https://example.com/low", priority=5)
        await queue.add_url("https://example.com/high", priority=0)
        await queue.add_url("https://example.com/mid", priority=2)

        self.assertEqual(await queue.get_next(), "https://example.com/high")
        queue.mark_processed("https://example.com/high")
        self.assertEqual(await queue.get_next(), "https://example.com/mid")
        queue.mark_processed("https://example.com/mid")
        self.assertEqual(await queue.get_next(), "https://example.com/low")

    async def test_queue_tracks_processed_failed_and_duplicates(self) -> None:
        queue = CrawlerQueue()

        self.assertTrue(await queue.add_url("https://example.com/page", depth=1))
        self.assertFalse(await queue.add_url("https://example.com/page", depth=1))
        self.assertEqual(queue.get_depth("https://example.com/page"), 1)
        next_url = await queue.get_next()
        self.assertEqual(next_url, "https://example.com/page")
        queue.mark_failed("https://example.com/page", "boom")

        stats = queue.get_stats()

        self.assertEqual(stats["failed"], 1)
        self.assertEqual(stats["processed"], 0)
        self.assertFalse(await queue.add_url("https://example.com/page", depth=1))


class SemaphoreManagerTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_semaphore_manager_tracks_active_tasks(self) -> None:
        manager = SemaphoreManager(max_concurrent=2, per_domain_concurrent=1)
        snapshots: list[dict[str, object]] = []

        async def worker(url: str) -> None:
            async with manager.limit(url):
                snapshots.append(manager.get_stats())
                await asyncio.sleep(0.01)

        await asyncio.gather(
            worker("https://example.com/1"),
            worker("https://example.com/2"),
            worker("https://other.example.org/1"),
        )

        self.assertTrue(any(int(snapshot["active_tasks"]) >= 1 for snapshot in snapshots))
        self.assertTrue(all(int(dict(snapshot["active_by_domain"]).get("example.com", 0)) <= 1 for snapshot in snapshots))
        self.assertEqual(manager.get_stats()["active_tasks"], 0)


class RateLimiterTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_rate_limiter_enforces_delay_for_same_domain(self) -> None:
        limiter = RateLimiter(requests_per_second=5.0, per_domain=True)

        started_at = time.perf_counter()
        await limiter.acquire("example.com")
        await limiter.acquire("example.com")
        elapsed = time.perf_counter() - started_at

        self.assertGreaterEqual(elapsed, 0.17)
        self.assertGreaterEqual(float(limiter.get_stats()["average_wait"]), 0.08)

    async def test_rate_limiter_keeps_domains_independent_when_per_domain_enabled(self) -> None:
        limiter = RateLimiter(requests_per_second=5.0, per_domain=True)

        await limiter.acquire("example.com")
        started_at = time.perf_counter()
        await limiter.acquire("other.example.org")
        elapsed = time.perf_counter() - started_at

        self.assertLess(elapsed, 0.08)

    async def test_rate_limiter_uses_global_limit_when_disabled_per_domain(self) -> None:
        limiter = RateLimiter(requests_per_second=5.0, per_domain=False)

        await limiter.acquire("example.com")
        started_at = time.perf_counter()
        await limiter.acquire("other.example.org")
        elapsed = time.perf_counter() - started_at

        self.assertGreaterEqual(elapsed, 0.17)


class RobotsParserTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.robots_txt = "User-agent: *\nDisallow:\n"
        app = web.Application()
        app.router.add_get("/robots.txt", self.handle_robots)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        sockets = self.site._server.sockets
        self.port = sockets[0].getsockname()[1]
        self.base_url = f"http://127.0.0.1:{self.port}"

    async def asyncTearDown(self) -> None:
        await self.runner.cleanup()

    async def handle_robots(self, request: web.Request) -> web.Response:
        return web.Response(text=self.robots_txt, content_type="text/plain")

    async def test_fetch_robots_parses_rules_and_caches_result(self) -> None:
        parser = RobotsParser(timeout=1.0, user_agent="MyBot/1.0")
        self.robots_txt = (
            "User-agent: MyBot\n"
            "Disallow: /private\n"
            "Allow: /private/public\n"
            "Crawl-delay: 0.25\n\n"
            "User-agent: *\n"
            "Disallow: /tmp\n"
        )

        rules = await parser.fetch_robots(self.base_url)
        can_fetch_private = parser.can_fetch(f"{self.base_url}/private/public", user_agent="MyBot/1.0")
        can_fetch_tmp = parser.can_fetch(f"{self.base_url}/tmp/file", user_agent="OtherBot/1.0")
        crawl_delay = parser.get_crawl_delay("MyBot/1.0")
        cached_rules = await parser.fetch_robots(self.base_url)

        self.assertTrue(dict(rules["rules"]).get("mybot") is not None)
        self.assertTrue(can_fetch_private)
        self.assertFalse(can_fetch_tmp)
        self.assertAlmostEqual(crawl_delay, 0.25, places=2)
        self.assertIs(rules, cached_rules)


class RetryStrategyTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_execute_with_retry_retries_transient_error_and_succeeds(self) -> None:
        strategy = RetryStrategy(max_retries=3, backoff_factor=2.0, base_delay=0.01)
        attempts = 0

        async def flaky_operation(url: str, _retry_attempt: int = 1) -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise TransientError("temporary failure", url=url)
            return "ok"

        result = await strategy.execute_with_retry(flaky_operation, "https://example.com")

        self.assertEqual(result, "ok")
        self.assertEqual(attempts, 3)
        self.assertEqual(strategy.get_stats()["successful_retries"], 1)

    async def test_execute_with_retry_does_not_retry_permanent_error(self) -> None:
        strategy = RetryStrategy(max_retries=3, backoff_factor=2.0, base_delay=0.01)
        attempts = 0

        async def permanent_failure(url: str) -> str:
            nonlocal attempts
            attempts += 1
            raise PermanentError("not found", url=url)

        with self.assertRaises(PermanentError):
            await strategy.execute_with_retry(permanent_failure, "https://example.com/404")

        self.assertEqual(attempts, 1)
        self.assertIn("https://example.com/404", list(strategy.get_stats()["permanent_error_urls"]))

    async def test_execute_with_retry_uses_exponential_backoff(self) -> None:
        strategy = RetryStrategy(max_retries=2, backoff_factor=2.0, base_delay=0.02)
        attempts = 0

        async def flaky_operation(url: str) -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise TransientError("busy", url=url, status_code=503)
            return "done"

        started_at = time.perf_counter()
        result = await strategy.execute_with_retry(flaky_operation, "https://example.com/503")
        elapsed = time.perf_counter() - started_at

        self.assertEqual(result, "done")
        self.assertGreaterEqual(elapsed, 0.05)
        self.assertGreaterEqual(float(strategy.get_stats()["average_retry_delay"]), 0.02)

    def test_classify_error_maps_network_and_parse_errors(self) -> None:
        strategy = RetryStrategy()

        network_error = strategy.classify_error(OSError("connection refused"), url="https://example.com")
        parse_error = strategy.record_error(ParseError("invalid html", url="https://example.com/page"), url="https://example.com/page")

        self.assertIsInstance(network_error, NetworkError)
        self.assertIsInstance(parse_error, ParseError)
        self.assertEqual(dict(strategy.get_stats()["error_counts_by_type"])["ParseError"], 1)


class StorageTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_json_storage_writes_jsonl_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "results.jsonl"
            storage = JSONStorage(str(file_path), buffer_size=2)

            await storage.save(
                {
                    "url": "https://example.com/1",
                    "title": "First",
                    "text": "Hello",
                    "links": ["https://example.com/2"],
                    "metadata": {"description": "demo"},
                    "status_code": 200,
                    "content_type": "text/html",
                }
            )
            await storage.save(
                {
                    "url": "https://example.com/2",
                    "title": "Second",
                    "text": "World",
                    "links": [],
                    "metadata": {},
                    "status_code": 200,
                    "content_type": "text/html",
                }
            )
            await storage.close()

            lines = [line for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            payload = [json.loads(line) for line in lines]
            self.assertEqual(len(payload), 2)
            self.assertEqual(payload[0]["url"], "https://example.com/1")
            self.assertEqual(storage.get_stats()["saved_count"], 2)

    async def test_csv_storage_writes_headers_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "results.csv"
            storage = CSVStorage(str(file_path), buffer_size=1)

            await storage.save(
                {
                    "url": "https://example.com/page",
                    "title": "Page, with comma",
                    "text": "Body with \"quotes\"",
                    "links": ["https://example.com/next"],
                    "metadata": {"lang": "ru"},
                    "status_code": 200,
                    "content_type": "text/html",
                }
            )
            await storage.close()

            with file_path.open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["url"], "https://example.com/page")
            self.assertIn("Page, with comma", rows[0]["title"])
            self.assertEqual(json.loads(rows[0]["metadata"])["lang"], "ru")

    async def test_sqlite_storage_inserts_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "results.sqlite3"
            storage = SQLiteStorage(str(db_path), batch_size=2)

            await storage.save(
                {
                    "url": "https://example.com/1",
                    "title": "First",
                    "text": "A",
                    "links": [],
                    "metadata": {},
                    "status_code": 200,
                    "content_type": "text/html",
                }
            )
            await storage.save(
                {
                    "url": "https://example.com/2",
                    "title": "Second",
                    "text": "B",
                    "links": ["https://example.com/3"],
                    "metadata": {"k": "v"},
                    "status_code": 200,
                    "content_type": "text/html",
                }
            )
            await storage.close()

            connection = sqlite3.connect(db_path)
            try:
                rows_count = int(connection.execute("SELECT COUNT(*) FROM pages").fetchone()[0])
                title = str(connection.execute("SELECT title FROM pages WHERE url = ?", ("https://example.com/2",)).fetchone()[0])
            finally:
                connection.close()

            self.assertEqual(rows_count, 2)
            self.assertEqual(title, "Second")

    async def test_storage_retries_write_error(self) -> None:
        class FlakyJSONStorage(JSONStorage):
            def __init__(self, file_path: str) -> None:
                super().__init__(file_path, buffer_size=1, retry_strategy=RetryStrategy(max_retries=1, base_delay=0.01))
                self.calls = 0

            async def _write_lines(self, records: list[dict[str, object]]) -> None:
                self.calls += 1
                if self.calls == 1:
                    raise OSError("temporary disk issue")
                await super()._write_lines(records)

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "results.jsonl"
            storage = FlakyJSONStorage(str(file_path))

            await storage.save(
                {
                    "url": "https://example.com/retry",
                    "title": "Retry",
                    "text": "ok",
                    "links": [],
                    "metadata": {},
                    "status_code": 200,
                    "content_type": "text/html",
                }
            )
            await storage.close()

            lines = [line for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(lines), 1)
            self.assertGreaterEqual(storage.retry_strategy.get_stats()["retry_attempts_total"], 1)


class AsyncCrawlerTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.robots_txt = "User-agent: *\nDisallow:\n"
        self.received_user_agents: list[str] = []
        self.request_times: dict[str, list[float]] = {}
        self.flaky_status_hits = 0
        app = web.Application()
        app.router.add_get("/robots.txt", self.handle_robots)
        app.router.add_get("/ok", self.handle_ok)
        app.router.add_get("/json", self.handle_json)
        app.router.add_get("/html", self.handle_html)
        app.router.add_get("/broken-html", self.handle_broken_html)
        app.router.add_get("/crawl/root", self.handle_crawl_root)
        app.router.add_get("/crawl/page-1", self.handle_crawl_page_1)
        app.router.add_get("/crawl/page-2", self.handle_crawl_page_2)
        app.router.add_get("/crawl/page-3", self.handle_crawl_page_3)
        app.router.add_get("/crawl/blocked", self.handle_crawl_blocked)
        app.router.add_get("/crawl/skip-me", self.handle_crawl_skip_me)
        app.router.add_get("/crawl/include-only", self.handle_crawl_include_only)
        app.router.add_get("/ua", self.handle_ua)
        app.router.add_get("/flaky", self.handle_flaky)
        app.router.add_get("/delay/{seconds}", self.handle_delay)
        app.router.add_get("/status/{code}", self.handle_status)

        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        sockets = self.site._server.sockets
        self.port = sockets[0].getsockname()[1]
        self.base_url = f"http://127.0.0.1:{self.port}"

    async def asyncTearDown(self) -> None:
        await self.runner.cleanup()

    def _record_request(self, request: web.Request, path: str) -> None:
        self.received_user_agents.append(request.headers.get("User-Agent", ""))
        self.request_times.setdefault(path, []).append(time.perf_counter())

    async def handle_robots(self, request: web.Request) -> web.Response:
        return web.Response(text=self.robots_txt, content_type="text/plain")

    async def handle_ok(self, request: web.Request) -> web.Response:
        self._record_request(request, "/ok")
        return web.Response(text="example page")

    async def handle_json(self, request: web.Request) -> web.Response:
        self._record_request(request, "/json")
        return web.json_response(
            {
                "args": {},
                "data": "",
                "files": {},
                "form": {},
                "headers": {
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Encoding": "deflate, gzip, br",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Host": "httpbin.org",
                    "User-Agent": "TelegramBot (like TwitterBot)",
                    "X-Amzn-Trace-Id": "Root=1-69e0ec2a-3157742401cfc9343c7862d2",
                },
                "origin": "149.154.161.218",
                "url": "https://httpbin.org/delay/1",
            }
        )

    async def handle_html(self, request: web.Request) -> web.Response:
        self._record_request(request, "/html")
        return web.Response(
            text=(
                "<html><head><title>Parser Page</title>"
                "<meta name='description' content='Parser description'>"
                "<meta name='keywords' content='parser, test'></head>"
                "<body><h1>Heading</h1><p>Body text</p><a href='/next'>Next</a>"
                "<img src='/image.png' alt='Image alt'><table><tr><td>A</td></tr></table>"
                "<ol><li>First</li></ol></body></html>"
            ),
            content_type="text/html",
        )

    async def handle_broken_html(self, request: web.Request) -> web.Response:
        self._record_request(request, "/broken-html")
        return web.Response(text="<html><head><title>Broken<title><body><a href='/oops'>Oops", content_type="text/html")

    async def handle_crawl_root(self, request: web.Request) -> web.Response:
        self._record_request(request, "/crawl/root")
        return web.Response(
            text=(
                f"<html><head><title>Root</title></head><body>"
                f"<a href='/crawl/page-1'>Page 1</a>"
                f"<a href='/crawl/page-2'>Page 2</a>"
                f"<a href='/crawl/page-1'>Page 1 Duplicate</a>"
                f"<a href='/crawl/include-only'>Include Only</a>"
                f"<a href='/crawl/blocked'>Blocked</a>"
                f"<a href='/crawl/skip-me'>Skip Me</a>"
                f"<a href='https://external.example.org/outside'>External</a>"
                f"</body></html>"
            ),
            content_type="text/html",
        )

    async def handle_crawl_page_1(self, request: web.Request) -> web.Response:
        self._record_request(request, "/crawl/page-1")
        return web.Response(
            text=(
                "<html><head><title>Page 1</title></head><body>"
                "<a href='/crawl/page-3'>Page 3</a>"
                "<a href='/crawl/page-2'>Page 2</a>"
                "</body></html>"
            ),
            content_type="text/html",
        )

    async def handle_crawl_page_2(self, request: web.Request) -> web.Response:
        self._record_request(request, "/crawl/page-2")
        return web.Response(
            text=(
                "<html><head><title>Page 2</title></head><body>"
                "<a href='/crawl/page-3'>Page 3</a>"
                "</body></html>"
            ),
            content_type="text/html",
        )

    async def handle_crawl_page_3(self, request: web.Request) -> web.Response:
        self._record_request(request, "/crawl/page-3")
        return web.Response(
            text="<html><head><title>Page 3</title></head><body><p>Leaf page</p></body></html>",
            content_type="text/html",
        )

    async def handle_crawl_blocked(self, request: web.Request) -> web.Response:
        self._record_request(request, "/crawl/blocked")
        return web.Response(text="<html><head><title>Blocked</title></head><body></body></html>", content_type="text/html")

    async def handle_crawl_skip_me(self, request: web.Request) -> web.Response:
        self._record_request(request, "/crawl/skip-me")
        return web.Response(text="<html><head><title>Skip</title></head><body></body></html>", content_type="text/html")

    async def handle_crawl_include_only(self, request: web.Request) -> web.Response:
        self._record_request(request, "/crawl/include-only")
        return web.Response(text="<html><head><title>Include</title></head><body></body></html>", content_type="text/html")

    async def handle_ua(self, request: web.Request) -> web.Response:
        self._record_request(request, "/ua")
        return web.Response(text=request.headers.get("User-Agent", ""))

    async def handle_flaky(self, request: web.Request) -> web.Response:
        self._record_request(request, "/flaky")
        self.flaky_status_hits += 1
        if self.flaky_status_hits == 1:
            return web.Response(status=500, text="temporary failure")
        return web.Response(text="recovered")

    async def handle_delay(self, request: web.Request) -> web.Response:
        self._record_request(request, "/delay")
        delay_seconds = float(request.match_info["seconds"])
        await asyncio.sleep(delay_seconds)
        return web.Response(text=f"delayed {delay_seconds}")

    async def handle_status(self, request: web.Request) -> web.Response:
        self._record_request(request, "/status")
        status_code = int(request.match_info["code"])
        return web.Response(status=status_code, text=f"status {status_code}")

    async def test_fetch_url_returns_text_for_valid_url(self) -> None:
        crawler = AsyncCrawler(max_concurrent=2)

        content = await crawler.fetch_url(f"{self.base_url}/ok")
        await crawler.close()

        self.assertEqual(content, "example page")

    async def test_fetch_urls_returns_all_requested_urls(self) -> None:
        crawler = AsyncCrawler(max_concurrent=3)
        urls = [
            f"{self.base_url}/ok",
            f"{self.base_url}/json",
            f"{self.base_url}/status/404",
        ]

        results = await crawler.fetch_urls(urls)
        await crawler.close()

        self.assertEqual(set(results.keys()), set(urls))
        self.assertIn("example page", results[f"{self.base_url}/ok"])
        self.assertIn('"origin": "149.154.161.218"', results[f"{self.base_url}/json"])
        self.assertEqual(results[f"{self.base_url}/status/404"], "")

    async def test_fetch_url_returns_empty_string_for_unreachable_url(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            unused_port = sock.getsockname()[1]
        crawler = AsyncCrawler(max_concurrent=1)

        content = await crawler.fetch_url(f"http://127.0.0.1:{unused_port}/missing")
        await crawler.close()

        self.assertEqual(content, "")

    async def test_fetch_url_returns_empty_string_on_timeout(self) -> None:
        crawler = AsyncCrawler(
            max_concurrent=1,
            read_timeout=0.05,
            total_timeout=0.1,
            retry_strategy=RetryStrategy(max_retries=1, base_delay=0.01),
        )

        content = await crawler.fetch_url(f"{self.base_url}/delay/0.2")
        retry_stats = crawler.retry_strategy.get_stats()
        await crawler.close()

        self.assertEqual(content, "")
        self.assertGreaterEqual(len(self.request_times.get("/delay", [])), 2)
        self.assertEqual(dict(crawler.error_details)[f"{self.base_url}/delay/0.2"]["type"], "TransientError")
        self.assertGreaterEqual(int(retry_stats["retry_attempts_total"]), 1)

    async def test_parallel_fetch_is_faster_than_sequential(self) -> None:
        urls = [f"{self.base_url}/delay/0.2?run={index}" for index in range(5)]
        parallel_crawler = AsyncCrawler(max_concurrent=5, requests_per_second=100.0, respect_robots=False, min_delay=0.0, jitter=0.0)
        sequential_crawler = AsyncCrawler(max_concurrent=1, requests_per_second=100.0, respect_robots=False, min_delay=0.0, jitter=0.0)

        parallel_started_at = time.perf_counter()
        parallel_results = await parallel_crawler.fetch_urls(urls)
        parallel_time = time.perf_counter() - parallel_started_at

        sequential_started_at = time.perf_counter()
        sequential_results = await fetch_urls_sequentially(sequential_crawler, urls)
        sequential_time = time.perf_counter() - sequential_started_at

        await parallel_crawler.close()
        await sequential_crawler.close()

        self.assertTrue(all(content for content in parallel_results.values()))
        self.assertTrue(all(content for content in sequential_results.values()))
        self.assertLess(parallel_time, sequential_time)

    async def test_logging_reports_start_success_and_error(self) -> None:
        crawler = AsyncCrawler(max_concurrent=2, retry_strategy=RetryStrategy(max_retries=1, base_delay=0.01))

        with self.assertLogs("src.models", level="INFO") as captured_logs:
            await crawler.fetch_url(f"{self.base_url}/ok")
            await crawler.fetch_url(f"{self.base_url}/status/500")
        await crawler.close()

        joined_logs = "\n".join(captured_logs.output)
        self.assertIn("Starting download", joined_logs)
        self.assertIn("Completed download", joined_logs)
        self.assertIn("Retry scheduled", joined_logs)

    async def test_fetch_and_parse_returns_parsed_page_data(self) -> None:
        crawler = AsyncCrawler(max_concurrent=2)

        result = await crawler.fetch_and_parse(f"{self.base_url}/html")
        await crawler.close()

        self.assertEqual(result["title"], "Parser Page")
        self.assertIn("Body text", str(result["text"]))
        self.assertEqual(list(result["links"]), [f"{self.base_url}/next"])
        self.assertEqual(list(result["images"]), [{"src": f"{self.base_url}/image.png", "alt": "Image alt"}])
        self.assertEqual(dict(result["metadata"])["description"], "Parser description")
        self.assertEqual(list(result["headings"]), [{"tag": "h1", "text": "Heading"}])
        self.assertEqual(list(result["tables"]), [[["A"]]])
        self.assertEqual(list(result["lists"]), [{"type": "ol", "items": ["First"]}])

    async def test_fetch_and_parse_returns_partial_result_for_invalid_page(self) -> None:
        crawler = AsyncCrawler(max_concurrent=2)

        result = await crawler.fetch_and_parse(f"{self.base_url}/status/404")
        await crawler.close()

        self.assertEqual(
            result,
            {
                "url": f"{self.base_url}/status/404",
                "title": "",
                "text": "",
                "links": [],
                "metadata": {},
                "images": [],
                "headings": [],
                "tables": [],
                "lists": [],
            },
        )

    async def test_fetch_and_parse_handles_broken_html(self) -> None:
        crawler = AsyncCrawler(max_concurrent=2)

        result = await crawler.fetch_and_parse(f"{self.base_url}/broken-html")
        await crawler.close()

        self.assertEqual(result["title"], "Broken")
        self.assertIn(f"{self.base_url}/oops", list(result["links"]))

    async def test_fetch_url_respects_min_delay(self) -> None:
        crawler = AsyncCrawler(
            max_concurrent=1,
            requests_per_second=100.0,
            respect_robots=False,
            min_delay=0.12,
            jitter=0.0,
        )

        started_at = time.perf_counter()
        await crawler.fetch_url(f"{self.base_url}/ok")
        await crawler.fetch_url(f"{self.base_url}/ok")
        elapsed = time.perf_counter() - started_at
        stats = crawler.rate_limiter.get_stats()
        await crawler.close()

        self.assertGreaterEqual(elapsed, 0.22)
        self.assertEqual(stats["acquire_count"], 2)

    async def test_fetch_url_respects_robots_crawl_delay(self) -> None:
        self.robots_txt = "User-agent: *\nCrawl-delay: 0.12\nDisallow:\n"
        crawler = AsyncCrawler(
            max_concurrent=1,
            requests_per_second=100.0,
            respect_robots=True,
            min_delay=0.0,
            jitter=0.0,
            user_agent="MyBot/1.0",
        )

        started_at = time.perf_counter()
        await crawler.fetch_url(f"{self.base_url}/ok")
        await crawler.fetch_url(f"{self.base_url}/ok")
        elapsed = time.perf_counter() - started_at
        await crawler.close()

        self.assertGreaterEqual(elapsed, 0.22)

    async def test_fetch_url_rotates_user_agents(self) -> None:
        crawler = AsyncCrawler(
            max_concurrent=1,
            requests_per_second=100.0,
            respect_robots=False,
            user_agent="MyBot/1.0",
            user_agents=["BotA/1.0", "BotB/1.0"],
        )

        first = await crawler.fetch_url(f"{self.base_url}/ua")
        second = await crawler.fetch_url(f"{self.base_url}/ua")
        await crawler.close()

        self.assertEqual(first, "BotA/1.0")
        self.assertEqual(second, "BotB/1.0")
        self.assertGreaterEqual(len(self.received_user_agents), 2)
        self.assertIn("BotA/1.0", self.received_user_agents)
        self.assertIn("BotB/1.0", self.received_user_agents)

    async def test_fetch_url_applies_backoff_after_server_error(self) -> None:
        crawler = AsyncCrawler(
            max_concurrent=1,
            requests_per_second=100.0,
            respect_robots=False,
            min_delay=0.0,
            jitter=0.0,
            backoff_base=0.12,
            backoff_factor=2.0,
            backoff_max=0.5,
            retry_strategy=RetryStrategy(max_retries=1, base_delay=0.01),
        )

        first = await crawler.fetch_url(f"{self.base_url}/flaky")
        self.flaky_status_hits = 0
        started_at = time.perf_counter()
        second = await crawler.fetch_url(f"{self.base_url}/flaky")
        elapsed = time.perf_counter() - started_at
        stats = crawler._get_crawl_stats(CrawlerQueue())
        await crawler.close()

        self.assertEqual(first, "recovered")
        self.assertEqual(second, "recovered")
        self.assertGreaterEqual(elapsed, 0.10)
        self.assertGreaterEqual(int(stats["backoff_count"]), 1)

    async def test_fetch_url_retries_500_and_succeeds(self) -> None:
        crawler = AsyncCrawler(
            max_concurrent=1,
            requests_per_second=100.0,
            respect_robots=False,
            min_delay=0.0,
            jitter=0.0,
            retry_strategy=RetryStrategy(max_retries=2, base_delay=0.01),
        )

        content = await crawler.fetch_url(f"{self.base_url}/flaky")
        retry_stats = crawler.retry_strategy.get_stats()
        await crawler.close()

        self.assertEqual(content, "recovered")
        self.assertEqual(len(self.request_times.get("/flaky", [])), 2)
        self.assertGreaterEqual(int(retry_stats["successful_retries"]), 1)
        self.assertGreaterEqual(int(retry_stats["retry_attempts_total"]), 1)

    async def test_fetch_url_does_not_retry_404(self) -> None:
        crawler = AsyncCrawler(
            max_concurrent=1,
            requests_per_second=100.0,
            respect_robots=False,
            min_delay=0.0,
            jitter=0.0,
            retry_strategy=RetryStrategy(max_retries=3, base_delay=0.01),
        )

        content = await crawler.fetch_url(f"{self.base_url}/status/404")
        retry_stats = crawler.retry_strategy.get_stats()
        await crawler.close()

        self.assertEqual(content, "")
        self.assertEqual(len(self.request_times.get("/status", [])), 1)
        self.assertEqual(dict(crawler.error_details)[f"{self.base_url}/status/404"]["type"], "PermanentError")
        self.assertEqual(int(retry_stats["retry_attempts_total"]), 0)

    async def test_fetch_and_parse_records_parse_error(self) -> None:
        class BrokenParser:
            async def parse_html(self, html: str, url: str) -> dict[str, object]:
                raise ValueError("cannot parse")

            def empty_result(self, url: str) -> dict[str, object]:
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

        crawler = AsyncCrawler(
            max_concurrent=1,
            requests_per_second=100.0,
            respect_robots=False,
            html_parser=BrokenParser(),
        )

        result = await crawler.fetch_and_parse(f"{self.base_url}/html")
        retry_stats = crawler.retry_strategy.get_stats()
        await crawler.close()

        self.assertEqual(result["url"], f"{self.base_url}/html")
        self.assertEqual(dict(crawler.error_details)[f"{self.base_url}/html"]["type"], "ParseError")
        self.assertEqual(dict(retry_stats["error_counts_by_type"])["ParseError"], 1)

    async def test_crawl_respects_max_depth(self) -> None:
        crawler = AsyncCrawler(max_concurrent=3, max_depth=1)

        result = await crawler.crawl(
            start_urls=[f"{self.base_url}/crawl/root"],
            max_pages=10,
            same_domain_only=True,
        )
        await crawler.close()

        processed_urls = dict(result["processed_urls"])
        self.assertIn(f"{self.base_url}/crawl/root", processed_urls)
        self.assertIn(f"{self.base_url}/crawl/page-1", processed_urls)
        self.assertIn(f"{self.base_url}/crawl/page-2", processed_urls)
        self.assertNotIn(f"{self.base_url}/crawl/page-3", processed_urls)
        self.assertEqual(processed_urls[f"{self.base_url}/crawl/root"]["depth"], 0)
        self.assertEqual(processed_urls[f"{self.base_url}/crawl/page-1"]["depth"], 1)

    async def test_crawl_filters_urls_and_skips_duplicates(self) -> None:
        crawler = AsyncCrawler(max_concurrent=3, max_depth=2)

        result = await crawler.crawl(
            start_urls=[f"{self.base_url}/crawl/root"],
            max_pages=10,
            same_domain_only=True,
            include_patterns=[r"include|page|root"],
            exclude_patterns=[r"skip-me"],
        )
        await crawler.close()

        processed_urls = dict(result["processed_urls"])
        visited_urls = list(result["visited_urls"])
        self.assertIn(f"{self.base_url}/crawl/include-only", processed_urls)
        self.assertNotIn(f"{self.base_url}/crawl/skip-me", processed_urls)
        self.assertEqual(visited_urls.count(f"{self.base_url}/crawl/page-1"), 1)
        self.assertTrue(all("external.example.org" not in url for url in processed_urls))

    async def test_crawl_collects_stats_and_state(self) -> None:
        crawler = AsyncCrawler(max_concurrent=2, max_depth=2, per_domain_concurrent=1, retry_strategy=RetryStrategy(max_retries=1, base_delay=0.01))

        with self.assertLogs("src.models", level="INFO") as captured_logs:
            result = await crawler.crawl(
                start_urls=[f"{self.base_url}/crawl/root", f"{self.base_url}/status/404"],
                max_pages=6,
                same_domain_only=True,
            )
        await crawler.close()

        stats = dict(result["stats"])
        self.assertGreaterEqual(int(stats["processed_pages"]), 1)
        self.assertGreaterEqual(int(stats["failed_pages"]), 1)
        self.assertIn(f"{self.base_url}/status/404", dict(result["failed_urls"]))
        self.assertIn("Crawl progress", "\n".join(captured_logs.output))
        self.assertEqual(dict(stats["semaphores"])["per_domain_concurrent"], 1)
        self.assertIn("retry", stats)

    async def test_crawl_blocks_urls_disallowed_by_robots(self) -> None:
        self.robots_txt = "User-agent: *\nDisallow: /crawl/blocked\n"
        crawler = AsyncCrawler(
            max_concurrent=2,
            max_depth=1,
            requests_per_second=100.0,
            respect_robots=True,
            user_agent="MyBot/1.0",
        )

        result = await crawler.crawl(
            start_urls=[f"{self.base_url}/crawl/root"],
            max_pages=10,
            same_domain_only=True,
        )
        await crawler.close()

        self.assertIn(f"{self.base_url}/crawl/blocked", dict(result["blocked_urls"]))
        self.assertNotIn(f"{self.base_url}/crawl/blocked", dict(result["processed_urls"]))
        self.assertEqual(len(self.request_times.get("/crawl/blocked", [])), 0)
        self.assertGreaterEqual(int(dict(result["stats"])["robots_blocked"]), 1)

    async def test_crawl_saves_processed_pages_to_json_storage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage_path = Path(tmp_dir) / "crawl.jsonl"
            storage = JSONStorage(str(storage_path), buffer_size=1)
            crawler = AsyncCrawler(
                max_concurrent=2,
                max_depth=1,
                requests_per_second=100.0,
                respect_robots=False,
                storage=storage,
            )

            result = await crawler.crawl(
                start_urls=[f"{self.base_url}/crawl/root"],
                max_pages=3,
                same_domain_only=True,
            )
            await crawler.close()

            lines = [line for line in storage_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            payload = [json.loads(line) for line in lines]
            self.assertGreaterEqual(len(payload), 1)
            self.assertEqual(payload[0]["status_code"], 200)
            self.assertIn("storage", dict(result["stats"]))

    async def test_crawl_continues_when_storage_save_fails(self) -> None:
        class BrokenStorage(JSONStorage):
            async def save(self, data: dict) -> None:
                raise OSError("disk full")

        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = BrokenStorage(str(Path(tmp_dir) / "broken.jsonl"))
            crawler = AsyncCrawler(
                max_concurrent=2,
                max_depth=1,
                requests_per_second=100.0,
                respect_robots=False,
                storage=storage,
            )

            result = await crawler.crawl(
                start_urls=[f"{self.base_url}/crawl/root"],
                max_pages=2,
                same_domain_only=True,
            )
            await crawler.close()

            self.assertGreaterEqual(len(dict(result["processed_urls"])), 1)
            self.assertGreaterEqual(len(dict(result["storage_errors"])), 1)
            self.assertGreaterEqual(int(dict(result["stats"])["storage_errors"]), 1)

    async def test_run_async_crawler_demo_returns_expected_shape(self) -> None:
        original_fetch_urls = AsyncCrawler.fetch_urls
        original_fetch_url = AsyncCrawler.fetch_url

        async def fake_fetch_urls(self, urls: list[str]) -> dict[str, str]:
            return {url: f"payload:{index}" for index, url in enumerate(urls)}

        async def fake_fetch_url(self, url: str) -> str:
            return f"payload:{url}"

        AsyncCrawler.fetch_urls = fake_fetch_urls
        AsyncCrawler.fetch_url = fake_fetch_url
        try:
            result = await run_async_crawler_demo()
        finally:
            AsyncCrawler.fetch_urls = original_fetch_urls
            AsyncCrawler.fetch_url = original_fetch_url

        self.assertEqual(len(list(result["urls"])), 6)
        self.assertEqual(set(dict(result["parallel_results"]).keys()), set(result["urls"]))
        self.assertEqual(set(dict(result["sequential_results"]).keys()), set(result["urls"]))
        self.assertGreaterEqual(float(result["parallel_elapsed"]), 0)
        self.assertGreaterEqual(float(result["sequential_elapsed"]), 0)

    async def test_run_html_parser_demo_writes_json_output(self) -> None:
        original_fetch_and_parse = AsyncCrawler.fetch_and_parse

        async def fake_fetch_and_parse(self, url: str) -> dict[str, object]:
            return {
                "url": url,
                "title": f"Title for {url}",
                "text": "hello world",
                "links": [f"{url}/next"],
                "metadata": {"title": f"Title for {url}", "description": "demo", "keywords": "test"},
                "images": [{"src": f"{url}/image.png", "alt": "alt"}],
                "headings": [{"tag": "h1", "text": "Heading"}],
                "tables": [[["col"]]],
                "lists": [{"type": "ul", "items": ["item"]}],
            }

        AsyncCrawler.fetch_and_parse = fake_fetch_and_parse
        try:
            result = await run_html_parser_demo()
        finally:
            AsyncCrawler.fetch_and_parse = original_fetch_and_parse

        output_path = Path(str(result["output_path"]))
        self.assertTrue(output_path.exists())
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(len(payload), 3)
        self.assertEqual(len(list(result["summary"])), 3)
        self.assertEqual(dict(enumerate(list(result["summary"])))[0]["links_count"], 1)
        self.assertEqual(dict(enumerate(list(result["summary"])))[0]["images_count"], 1)

    async def test_run_site_crawl_demo_writes_json_output(self) -> None:
        original_crawl = AsyncCrawler.crawl

        async def fake_crawl(self, start_urls: list[str], max_pages: int = 100, same_domain_only: bool = False, include_patterns=None, exclude_patterns=None, max_depth=None) -> dict[str, object]:
            return {
                "processed_urls": {
                    start_urls[0]: {
                        "url": start_urls[0],
                        "title": "Demo page",
                        "text": "demo",
                        "links": [f"{start_urls[0]}/next"],
                        "metadata": {},
                        "images": [],
                        "headings": [],
                        "tables": [],
                        "lists": [],
                        "depth": 0,
                    }
                },
                "failed_urls": {},
                "visited_urls": start_urls,
                "stats": {"pending": 0, "processed_pages": 1, "failed_pages": 0, "pages_per_second": 5.0},
            }

        AsyncCrawler.crawl = fake_crawl
        try:
            result = await run_site_crawl_demo()
        finally:
            AsyncCrawler.crawl = original_crawl

        output_path = Path(str(result["output_path"]))
        self.assertTrue(output_path.exists())
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(list(payload["processed_urls"].keys()), ["https://example.com"])
        self.assertEqual(dict(result["stats"])["processed_pages"], 1)

    async def test_run_polite_crawl_demo_writes_json_output(self) -> None:
        original_crawl = AsyncCrawler.crawl

        async def fake_crawl(self, start_urls: list[str], max_pages: int = 100, same_domain_only: bool = False, include_patterns=None, exclude_patterns=None, max_depth=None) -> dict[str, object]:
            return {
                "processed_urls": {},
                "failed_urls": {},
                "blocked_urls": {f"{start_urls[0]}/admin": "blocked by robots.txt"},
                "visited_urls": start_urls,
                "stats": {"processed_pages": 0, "failed_pages": 0, "robots_blocked": 1, "current_req_per_sec": 1.5, "average_delay": 0.5, "backoff_count": 0},
            }

        AsyncCrawler.crawl = fake_crawl
        try:
            result = await run_polite_crawl_demo()
        finally:
            AsyncCrawler.crawl = original_crawl

        output_path = Path(str(result["output_path"]))
        self.assertTrue(output_path.exists())
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(dict(payload["blocked_urls"])["https://example.com/admin"], "blocked by robots.txt")
        self.assertEqual(dict(result["stats"])["robots_blocked"], 1)

    async def test_run_retry_demo_writes_json_output(self) -> None:
        original_fetch_urls = AsyncCrawler.fetch_urls

        async def fake_fetch_urls(self, urls: list[str]) -> dict[str, str]:
            self.error_details = {urls[1]: {"type": "TransientError", "message": "HTTP 503: busy"}}
            self.retry_strategy._retry_attempts_total = 2
            self.retry_strategy._successful_retries = 1
            self.retry_strategy._total_retry_delay = 0.3
            self.retry_strategy._error_counts_by_type = {"TransientError": 1, "PermanentError": 1}
            self.retry_strategy._permanent_error_urls = {urls[2]}
            return {urls[0]: "ok", urls[1]: "", urls[2]: ""}

        AsyncCrawler.fetch_urls = fake_fetch_urls
        try:
            result = await run_retry_demo()
        finally:
            AsyncCrawler.fetch_urls = original_fetch_urls

        output_path = Path(str(result["output_path"]))
        self.assertTrue(output_path.exists())
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(dict(payload["error_details"])["https://httpbin.org/status/503"]["type"], "TransientError")
        self.assertEqual(dict(result["retry_stats"])["successful_retries"], 1)

    async def test_run_storage_demo_returns_saved_counts(self) -> None:
        original_crawl = AsyncCrawler.crawl

        async def fake_crawl(self, start_urls: list[str], max_pages: int = 100, same_domain_only: bool = False, include_patterns=None, exclude_patterns=None, max_depth=None) -> dict[str, object]:
            sample_record = {
                "url": start_urls[0],
                "title": "Stored page",
                "text": "body",
                "links": [],
                "metadata": {},
                "crawled_at": datetime.now(timezone.utc).isoformat(),
                "status_code": 200,
                "content_type": "text/html",
            }
            if self.storage is not None:
                await self.storage.save(sample_record)
            return {
                "processed_urls": {start_urls[0]: {"url": start_urls[0], "title": "Stored page", "text": "body", "links": [], "metadata": {}, "images": [], "headings": [], "tables": [], "lists": [], "depth": 0}},
                "failed_urls": {},
                "blocked_urls": {},
                "error_details": {},
                "storage_errors": {},
                "visited_urls": start_urls,
                "stats": {"processed_pages": 1, "failed_pages": 0, "storage": self.storage.get_stats() if self.storage is not None else {}},
            }

        AsyncCrawler.crawl = fake_crawl
        try:
            result = await run_storage_demo()
        finally:
            AsyncCrawler.crawl = original_crawl

        self.assertTrue(Path(str(result["json_output_path"])).exists())
        self.assertTrue(Path(str(result["csv_output_path"])).exists())
        self.assertTrue(Path(str(result["sqlite_output_path"])).exists())
        self.assertGreaterEqual(int(result["json_records"]), 1)
        self.assertGreaterEqual(int(result["csv_records"]), 1)
        self.assertGreaterEqual(int(result["sqlite_rows"]), 1)


if __name__ == "__main__":
    unittest.main()
