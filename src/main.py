from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
from typing import Callable

try:
    from .models import (
        AbstractAccount,
        Bank,
        Client,
        InvalidOperationError,
        ReportBuilder,
        RiskLevel,
        Transaction,
        TransactionProcessor,
        TransactionQueue,
        TransactionStatus,
        TransactionType,
    )
except ImportError:
    from models import (
        AbstractAccount,
        Bank,
        Client,
        InvalidOperationError,
        ReportBuilder,
        RiskLevel,
        Transaction,
        TransactionProcessor,
        TransactionQueue,
        TransactionStatus,
        TransactionType,
    )


def print_section(title: str) -> None:
    line = "=" * 88
    print(f"\n{line}\n{title}\n{line}")


def mask_identifier(value: str) -> str:
    if len(value) <= 6:
        return value
    return f"{value[:4]}...{value[-2:]}"


def format_currency_map(values: dict[str, object]) -> str:
    formatted_parts = []
    for currency_code, amount in values.items():
        formatted_parts.append(f"{currency_code}: {amount}")
    return ", ".join(formatted_parts)


def run_during_allowed_hours(bank: Bank, action: Callable[[], object]) -> object:
    original_current_hour = bank._current_hour
    bank._current_hour = lambda current_time=None: 10 if current_time is None else current_time.hour
    try:
        return action()
    finally:
        bank._current_hour = original_current_hour


def build_demo_bank() -> tuple[
    Bank,
    TransactionProcessor,
    TransactionQueue,
    dict[str, Client],
    dict[str, AbstractAccount],
    dict[str, datetime],
    Path,
]:
    started_at = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    morning_time = started_at + timedelta(minutes=1)
    midday_time = morning_time + timedelta(hours=1)
    night_time = (morning_time + timedelta(days=1)).replace(hour=1, minute=30, second=0, microsecond=0)
    audit_log_path = Path(tempfile.gettempdir()) / f"dj_vy_demo_audit_{started_at.strftime('%Y%m%d_%H%M%S')}.jsonl"
    bank = Bank(name="Cascade Demo Bank", audit_log_file_path=str(audit_log_path))

    clients_data = {
        "alice": ("Alice Morozova", "alice.morozova@example.com", "+77001110001", 29, "1024"),
        "boris": ("Boris Sokolov", "boris.sokolov@example.com", "+77001110002", 35, "2048"),
        "clara": ("Clara Kim", "clara.kim@example.com", "+77001110003", 31, "4096"),
        "dmitry": ("Dmitry Orlov", "dmitry.orlov@example.com", "+77001110004", 38, "5120"),
        "eva": ("Eva Petrova", "eva.petrova@example.com", "+77001110005", 27, "6144"),
        "farid": ("Farid Nurgali", "farid.nurgali@example.com", "+77001110006", 33, "7168"),
        "gina": ("Gina Zhao", "gina.zhao@example.com", "+77001110007", 30, "8192"),
    }
    clients: dict[str, Client] = {}
    for key, client_data in clients_data.items():
        client = Client(
            full_name=client_data[0],
            email=client_data[1],
            phone=client_data[2],
            age=client_data[3],
            pin_code=client_data[4],
        )
        bank.add_client(client)
        clients[key] = client

    accounts: dict[str, AbstractAccount] = {}

    def open_account(key: str, client_key: str, account_type: str, **kwargs: object) -> None:
        account = run_during_allowed_hours(
            bank,
            lambda: bank.open_account(clients[client_key].client_id, account_type=account_type, **kwargs),
        )
        accounts[key] = account

    open_account("alice_rub_main", "alice", "bank", balance="5000.00", currency="RUB")
    open_account(
        "alice_usd_savings",
        "alice",
        "savings",
        balance="1200.00",
        currency="USD",
        min_balance="100.00",
        monthly_interest_rate="0.01",
    )
    open_account(
        "alice_rub_premium",
        "alice",
        "premium",
        balance="1800.00",
        currency="RUB",
        overdraft_limit="700.00",
        fixed_commission="5.00",
    )
    open_account("boris_rub_main", "boris", "bank", balance="2400.00", currency="RUB")
    open_account(
        "boris_usd_invest",
        "boris",
        "investment",
        balance="300.00",
        currency="USD",
        portfolio={"stocks": "500.00", "bonds": "250.00", "etf": "50.00"},
    )
    open_account(
        "clara_eur_premium",
        "clara",
        "premium",
        balance="2600.00",
        currency="EUR",
        overdraft_limit="500.00",
        fixed_commission="7.00",
    )
    open_account("clara_rub_main", "clara", "bank", balance="900.00", currency="RUB")
    open_account("dmitry_usd_main", "dmitry", "bank", balance="2500.00", currency="USD")
    open_account(
        "dmitry_kzt_savings",
        "dmitry",
        "savings",
        balance="300000.00",
        currency="KZT",
        min_balance="25000.00",
        monthly_interest_rate="0.005",
    )
    open_account("dmitry_rub_main", "dmitry", "bank", balance="700.00", currency="RUB")
    open_account("eva_eur_main", "eva", "bank", balance="2100.00", currency="EUR")
    open_account(
        "eva_rub_invest",
        "eva",
        "investment",
        balance="600.00",
        currency="RUB",
        portfolio={"stocks": "1000.00", "bonds": "300.00", "etf": "100.00"},
    )
    open_account(
        "farid_rub_premium",
        "farid",
        "premium",
        balance="1600.00",
        currency="RUB",
        overdraft_limit="900.00",
        fixed_commission="8.00",
    )
    open_account("gina_cny_main", "gina", "bank", balance="8000.00", currency="CNY")
    open_account(
        "gina_rub_savings",
        "gina",
        "savings",
        balance="1800.00",
        currency="RUB",
        min_balance="200.00",
        monthly_interest_rate="0.004",
    )

    accounts["alice_usd_savings"].apply_monthly_interest()
    accounts["gina_rub_savings"].apply_monthly_interest()

    processor = TransactionProcessor(
        bank,
        exchange_rates={
            ("RUB", "USD"): "0.0110",
            ("USD", "RUB"): "91.0000",
            ("USD", "EUR"): "0.9200",
            ("EUR", "USD"): "1.0870",
            ("EUR", "RUB"): "98.0000",
            ("RUB", "KZT"): "5.3000",
            ("USD", "CNY"): "7.1000",
            ("CNY", "EUR"): "0.1270",
        },
        external_transfer_fee="10.00",
        max_retries=3,
    )
    queue = TransactionQueue()
    phase_times = {
        "morning": morning_time,
        "midday": midday_time,
        "night": night_time,
    }
    return bank, processor, queue, clients, accounts, phase_times, audit_log_path


def build_demo_transactions(accounts: dict[str, AbstractAccount], phase_times: dict[str, datetime]) -> dict[str, Transaction]:
    transactions: dict[str, Transaction] = {}

    def register(
        label: str,
        transaction_type: TransactionType,
        amount: str,
        currency: str,
        sender_key: str,
        recipient_key: str,
        scheduled_at: datetime | None = None,
    ) -> None:
        transactions[label] = Transaction(
            transaction_type=transaction_type,
            amount=amount,
            currency=currency,
            sender_account_id=accounts[sender_key].account_id,
            recipient_account_id=accounts[recipient_key].account_id,
            scheduled_at=scheduled_at,
        )

    register("T01", TransactionType.TRANSFER_EXTERNAL, "150.00", "RUB", "alice_rub_main", "boris_rub_main")
    register("T02", TransactionType.TRANSFER_EXTERNAL, "80.00", "EUR", "eva_eur_main", "alice_usd_savings")
    register("T03", TransactionType.EXCHANGE, "200.00", "RUB", "alice_rub_premium", "alice_usd_savings")
    register("T04", TransactionType.EXCHANGE, "60.00", "USD", "boris_usd_invest", "boris_rub_main")
    register("T05", TransactionType.EXCHANGE, "120.00", "EUR", "clara_eur_premium", "clara_rub_main")
    register("T06", TransactionType.TRANSFER_EXTERNAL, "90.00", "USD", "dmitry_usd_main", "gina_cny_main")
    register("T07", TransactionType.TRANSFER_EXTERNAL, "160.00", "RUB", "gina_rub_savings", "alice_rub_main")
    register("T08", TransactionType.TRANSFER_EXTERNAL, "110.00", "RUB", "farid_rub_premium", "clara_rub_main")
    register("T09", TransactionType.TRANSFER_EXTERNAL, "140.00", "RUB", "eva_rub_invest", "boris_rub_main")
    register("T10", TransactionType.TRANSFER_EXTERNAL, "130.00", "RUB", "boris_rub_main", "gina_rub_savings")
    register("T11", TransactionType.TRANSFER_EXTERNAL, "80.00", "RUB", "dmitry_rub_main", "farid_rub_premium")
    register("T12", TransactionType.EXCHANGE, "50.00", "USD", "alice_usd_savings", "alice_rub_main")
    register("T13", TransactionType.TRANSFER_EXTERNAL, "60.00", "RUB", "clara_rub_main", "dmitry_rub_main")
    register("T14", TransactionType.TRANSFER_EXTERNAL, "300.00", "CNY", "gina_cny_main", "eva_eur_main")
    register("T15", TransactionType.TRANSFER_EXTERNAL, "200.00", "RUB", "boris_rub_main", "alice_rub_main")
    register("T16", TransactionType.TRANSFER_EXTERNAL, "95.00", "EUR", "eva_eur_main", "clara_eur_premium")
    register("T17", TransactionType.TRANSFER_EXTERNAL, "70.00", "USD", "dmitry_usd_main", "eva_eur_main")
    register("T18", TransactionType.TRANSFER_EXTERNAL, "125.00", "RUB", "alice_rub_main", "gina_rub_savings")
    register("T19", TransactionType.TRANSFER_EXTERNAL, "90.00", "RUB", "farid_rub_premium", "boris_rub_main")
    register("T20", TransactionType.TRANSFER_EXTERNAL, "75.00", "RUB", "gina_rub_savings", "farid_rub_premium")
    register("T21", TransactionType.TRANSFER_EXTERNAL, "1300.00", "RUB", "alice_rub_main", "boris_rub_main")
    register("T22", TransactionType.TRANSFER_EXTERNAL, "1500.00", "EUR", "eva_eur_main", "alice_usd_savings")
    register("T23", TransactionType.TRANSFER_EXTERNAL, "1250.00", "RUB", "boris_rub_main", "gina_rub_savings")
    register("T24", TransactionType.EXCHANGE, "1100.00", "EUR", "clara_eur_premium", "clara_rub_main")
    register("T25", TransactionType.TRANSFER_EXTERNAL, "2200.00", "CNY", "gina_cny_main", "eva_eur_main")
    register("T26", TransactionType.TRANSFER_EXTERNAL, "1300.00", "USD", "dmitry_usd_main", "gina_cny_main")
    register("T27", TransactionType.TRANSFER_INTERNAL, "50.00", "RUB", "alice_rub_main", "boris_rub_main")
    register("T28", TransactionType.TRANSFER_EXTERNAL, "40.00", "RUB", "boris_rub_main", "boris_usd_invest")
    register("T29", TransactionType.EXCHANGE, "60.00", "EUR", "clara_eur_premium", "dmitry_usd_main")
    register("T30", TransactionType.TRANSFER_EXTERNAL, "5000.00", "RUB", "dmitry_rub_main", "farid_rub_premium")
    register(
        "T31",
        TransactionType.TRANSFER_EXTERNAL,
        "35.00",
        "USD",
        "alice_rub_main",
        "gina_rub_savings",
        scheduled_at=phase_times["midday"],
    )
    register(
        "T32",
        TransactionType.TRANSFER_EXTERNAL,
        "45.00",
        "RUB",
        "alice_rub_main",
        "dmitry_rub_main",
        scheduled_at=phase_times["midday"],
    )
    register(
        "T33",
        TransactionType.TRANSFER_EXTERNAL,
        "70.00",
        "USD",
        "dmitry_usd_main",
        "eva_eur_main",
        scheduled_at=phase_times["midday"],
    )
    register(
        "T34",
        TransactionType.TRANSFER_EXTERNAL,
        "65.00",
        "RUB",
        "farid_rub_premium",
        "gina_rub_savings",
        scheduled_at=phase_times["midday"],
    )
    register(
        "T35",
        TransactionType.TRANSFER_EXTERNAL,
        "5000.00",
        "RUB",
        "eva_rub_invest",
        "alice_rub_main",
        scheduled_at=phase_times["midday"],
    )
    register(
        "T36",
        TransactionType.TRANSFER_EXTERNAL,
        "1800.00",
        "RUB",
        "alice_rub_main",
        "eva_eur_main",
        scheduled_at=phase_times["night"],
    )
    register(
        "T37",
        TransactionType.TRANSFER_EXTERNAL,
        "1600.00",
        "RUB",
        "boris_rub_main",
        "farid_rub_premium",
        scheduled_at=phase_times["night"],
    )
    register(
        "T38",
        TransactionType.TRANSFER_EXTERNAL,
        "50.00",
        "RUB",
        "gina_rub_savings",
        "farid_rub_premium",
        scheduled_at=phase_times["night"],
    )
    register(
        "T39",
        TransactionType.TRANSFER_EXTERNAL,
        "45.00",
        "RUB",
        "alice_rub_main",
        "boris_rub_main",
        scheduled_at=phase_times["night"],
    )
    register(
        "T40",
        TransactionType.TRANSFER_EXTERNAL,
        "65.00",
        "RUB",
        "gina_rub_savings",
        "alice_rub_main",
        scheduled_at=phase_times["night"],
    )
    return transactions


def describe_transaction(transaction: Transaction, bank: Bank) -> str:
    sender_client = bank.clients[bank.account_to_client[transaction.sender_account_id]].full_name
    recipient_client = bank.clients[bank.account_to_client[transaction.recipient_account_id]].full_name
    return (
        f"{transaction.transaction_id} | {transaction.transaction_type.value} | "
        f"{transaction.amount:.2f} {transaction.currency.value} | "
        f"{sender_client} ({mask_identifier(transaction.sender_account_id)}) -> "
        f"{recipient_client} ({mask_identifier(transaction.recipient_account_id)})"
    )


def risk_trace(bank: Bank, transaction_id: str) -> str:
    events = bank.audit_log.filter_events(event_type="risk_assessment", transaction_id=transaction_id)
    if not events:
        return ""
    event = events[-1]
    if event.risk_level is None:
        return ""
    return f" | risk={event.risk_level.value} | reasons={event.message}"


def enqueue_transactions(queue: TransactionQueue, bank: Bank, transactions: dict[str, Transaction]) -> None:
    print_section("QUEUE LOADING")
    for label, transaction in transactions.items():
        queue.add_transaction(transaction)
        scheduled_text = (
            f" scheduled for {transaction.scheduled_at.isoformat()}" if transaction.scheduled_at is not None else " immediate"
        )
        print(f"[QUEUE] {label} -> {describe_transaction(transaction, bank)} | status={transaction.status.value}{scheduled_text}")


def process_phase(label: str, queue: TransactionQueue, processor: TransactionProcessor, bank: Bank, now: datetime) -> list[Transaction]:
    print_section(f"PROCESSING PHASE: {label} @ {now.isoformat()}")
    ready_transactions = queue.get_ready_transactions(now)
    if not ready_transactions:
        print("No transactions are ready for processing.")
        return []
    print(f"Ready in queue: {len(ready_transactions)}")
    processed_transactions = processor.process_queue(queue, now)
    for transaction in processed_transactions:
        if transaction.status == TransactionStatus.COMPLETED:
            print(f"[OK] {describe_transaction(transaction, bank)} | fee={transaction.fee:.2f}{risk_trace(bank, transaction.transaction_id)}")
            continue
        if transaction.status == TransactionStatus.FAILED:
            print(
                f"[REJECTED] {describe_transaction(transaction, bank)} | reason={transaction.failure_reason}"
                f"{risk_trace(bank, transaction.transaction_id)}"
            )
            continue
        if transaction.status == TransactionStatus.RETRYING:
            print(f"[RETRY] {describe_transaction(transaction, bank)} | error={transaction.error_log[-1]}")
    return processed_transactions


def show_client_scenarios(bank: Bank, clients: dict[str, Client], transactions: dict[str, Transaction]) -> None:
    print_section("CLIENT SCENARIOS")
    for client_key in ("alice", "boris", "gina"):
        client = clients[client_key]
        print(f"\nClient: {client.full_name} | status={client.status.value} | client_id={client.client_id}")
        print("Accounts:")
        for account_id in client.account_ids:
            account = bank.accounts[account_id]
            account_info = account.get_account_info()
            print(
                f"  - {account_info['account_type']} | {account_info['account_id']} | "
                f"{account_info['status']} | {account_info['balance']} {account_info['currency']}"
            )
        print("Transaction history:")
        client_transactions = [
            transaction
            for transaction in transactions.values()
            if bank.account_to_client[transaction.sender_account_id] == client.client_id
            or bank.account_to_client[transaction.recipient_account_id] == client.client_id
        ]
        for transaction in client_transactions[:8]:
            print(
                f"  - {transaction.transaction_id} | {transaction.transaction_type.value} | "
                f"{transaction.status.value} | {transaction.amount:.2f} {transaction.currency.value}"
            )
        print("Audit history:")
        for event in bank.audit_log.filter_events(client_id=client.client_id)[-6:]:
            risk_level = event.risk_level.value if event.risk_level is not None else "n/a"
            print(f"  - {event.timestamp.isoformat()} | {event.event_type} | severity={event.severity.value} | risk={risk_level}")
        print("Suspicious activity:")
        if not client.suspicious_activity:
            print("  - none")
        else:
            for event in client.suspicious_activity:
                print(f"  - {event}")
        risk_profile = bank.get_client_risk_profile(client.client_id)
        print(f"Risk profile: {risk_profile}")


def demonstrate_security_scenarios(bank: Bank, clients: dict[str, Client]) -> None:
    print_section("SECURITY INCIDENTS")
    target_client = clients["farid"]
    for attempt in range(1, 4):
        try:
            bank.authenticate_client(target_client.client_id, "0000")
        except InvalidOperationError as exc:
            print(
                f"[AUTH FAILED] attempt={attempt} | client={target_client.full_name} | "
                f"reason={str(exc)} | failed_attempts={target_client.failed_auth_attempts}"
            )
    try:
        bank.authenticate_client(target_client.client_id, target_client.pin_code)
    except InvalidOperationError as exc:
        print(f"[AUTH BLOCKED] client={target_client.full_name} | reason={str(exc)}")
    target_account_id = next(iter(target_client.account_ids))
    try:
        run_during_allowed_hours(bank, lambda: bank.freeze_account(target_account_id))
    except InvalidOperationError as exc:
        print(f"[BLOCKED ACTION] client={target_client.full_name} | reason={str(exc)}")


def generate_report_artifacts(
    bank: Bank,
    clients: dict[str, Client],
    audit_log_path: Path,
    client_keys: tuple[str, ...] | None = None,
) -> dict[str, object]:
    report_builder = ReportBuilder(bank)
    output_dir = audit_log_path.parent / f"{audit_log_path.stem}_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_client_keys = tuple(client_keys) if client_keys is not None else tuple(list(clients.keys())[:3])
    client_reports: dict[str, dict[str, str]] = {}
    for client_key in selected_client_keys:
        client_report = report_builder.build_client_report(clients[client_key].client_id)
        client_reports[client_key] = {
            "json": report_builder.export_to_json(client_report, str(output_dir / f"{client_key}_client_report.json")),
            "csv": report_builder.export_to_csv(client_report, str(output_dir / f"{client_key}_client_report.csv")),
        }
    bank_report = report_builder.build_bank_report()
    risk_report = report_builder.build_risk_report()
    bank_report_files = {
        "json": report_builder.export_to_json(bank_report, str(output_dir / "bank_report.json")),
        "csv": report_builder.export_to_csv(bank_report, str(output_dir / "bank_report.csv")),
    }
    risk_report_files = {
        "json": report_builder.export_to_json(risk_report, str(output_dir / "risk_report.json")),
        "csv": report_builder.export_to_csv(risk_report, str(output_dir / "risk_report.csv")),
    }
    charts = report_builder.save_charts(
        str(output_dir / "charts"),
        client_id=clients[selected_client_keys[0]].client_id if selected_client_keys else None,
    )
    return {
        "output_dir": str(output_dir),
        "client_reports": client_reports,
        "bank_report": bank_report_files,
        "risk_report": risk_report_files,
        "risk_preview": report_builder.render_text(risk_report),
        "charts": charts,
    }


def show_reports(
    bank: Bank,
    transactions: dict[str, Transaction],
    audit_log_path: Path,
    report_artifacts: dict[str, object],
) -> None:
    print_section("SUSPICIOUS OPERATIONS REPORT")
    suspicious_operations = bank.get_audit_report_suspicious_operations(RiskLevel.MEDIUM)
    for event in suspicious_operations[:12]:
        print(
            f"- {event['timestamp']} | tx={event['transaction_id']} | risk={event['risk_level']} | "
            f"message={event['message']}"
        )
    print(f"Total suspicious operations in report: {len(suspicious_operations)}")

    print_section("TOP-3 CLIENTS BY CURRENCY")
    rankings = bank.get_clients_ranking()
    for currency_code, ranking in rankings.items():
        if not ranking:
            continue
        print(f"{currency_code}:")
        for position, item in enumerate(ranking[:3], start=1):
            print(
                f"  {position}. {item['full_name']} | status={item['status']} | "
                f"assets={item['total_assets']} | accounts={item['accounts_count']}"
            )

    print_section("TRANSACTION STATISTICS")
    status_counts = Counter(transaction.status.value for transaction in transactions.values())
    type_counts = Counter(transaction.transaction_type.value for transaction in transactions.values())
    print(f"By status: {dict(status_counts)}")
    print(f"By type: {dict(type_counts)}")
    print(f"Audit error statistics: {bank.get_audit_error_statistics()}")

    print_section("BANK TOTAL BALANCE")
    total_balance = bank.get_total_balance()
    print(format_currency_map({currency_code: f"{amount:.2f}" for currency_code, amount in total_balance.items()}))

    print_section("REPORT EXPORTS")
    print(f"Artifacts directory: {report_artifacts['output_dir']}")
    print("Client reports:")
    for client_key, file_set in dict(report_artifacts["client_reports"]).items():
        print(f"  - {client_key}: json={file_set['json']} | csv={file_set['csv']}")
    bank_report_files = dict(report_artifacts["bank_report"])
    risk_report_files = dict(report_artifacts["risk_report"])
    print(f"Bank report: json={bank_report_files['json']} | csv={bank_report_files['csv']}")
    print(f"Risk report: json={risk_report_files['json']} | csv={risk_report_files['csv']}")
    print(f"Charts saved: {len(list(report_artifacts['charts']))}")
    for chart_path in list(report_artifacts["charts"]):
        print(f"  - {chart_path}")
    preview_lines = str(report_artifacts["risk_preview"]).splitlines()
    print("Risk report text preview:")
    for line in preview_lines[:8]:
        print(f"  {line}")

    print_section("DEMO SUMMARY")
    print(f"Clients: {len(bank.clients)}")
    print(f"Accounts: {len(bank.accounts)}")
    print(f"Transactions: {len(transactions)}")
    print(f"Audit events captured: {len(bank.audit_log.events)}")
    print(f"Audit JSONL file: {audit_log_path}")


def main() -> None:
    print_section("BANKING SYSTEM DEMONSTRATION")
    bank, processor, queue, clients, accounts, phase_times, audit_log_path = build_demo_bank()
    print(f"Bank initialized: {bank.name}")
    print(f"Clients created: {len(clients)}")
    print(f"Accounts opened: {len(accounts)}")
    print(f"Processing phases: {', '.join(f'{name}={moment.isoformat()}' for name, moment in phase_times.items())}")

    transactions = build_demo_transactions(accounts, phase_times)
    enqueue_transactions(queue, bank, transactions)

    cancelled_transaction = queue.cancel_transaction(transactions["T33"].transaction_id, "client changed mind before settlement")
    print(f"[CANCELLED] T33 -> {describe_transaction(cancelled_transaction, bank)} | reason={cancelled_transaction.failure_reason}")

    process_phase("Morning batch", queue, processor, bank, phase_times["morning"])

    run_during_allowed_hours(bank, lambda: bank.freeze_account(accounts["dmitry_rub_main"].account_id))
    print(f"[SECURITY] Frozen account for scenario: {mask_identifier(accounts['dmitry_rub_main'].account_id)}")

    process_phase("Midday batch", queue, processor, bank, phase_times["midday"])
    process_phase("Night batch", queue, processor, bank, phase_times["night"])

    demonstrate_security_scenarios(bank, clients)
    report_artifacts = generate_report_artifacts(bank, clients, audit_log_path)
    show_client_scenarios(bank, clients, transactions)
    show_reports(bank, transactions, audit_log_path, report_artifacts)


if __name__ == "__main__":
    main()
