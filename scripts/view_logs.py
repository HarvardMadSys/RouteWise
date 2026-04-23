#!/usr/bin/env python3
"""View and analyze logs from the OpenRouter database."""

import json
import sqlite3

from tabulate import tabulate


def connect_db(db_path: str = "var/db/openrouter_logs.db") -> sqlite3.Connection:
    """Connect to the SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def show_recent_logs(conn: sqlite3.Connection, limit: int = 10) -> None:
    """Show recent API requests."""
    print("\n Recent API Requests")
    print("=" * 100)

    cursor = conn.execute(
        """
        SELECT
            substr(request_id, 5, 8) as req_id,
            datetime(timestamp) as time,
            model_id,
            provider,
            prompt_tokens as p_tok,
            completion_tokens as c_tok,
            total_tokens as total,
            latency_ms as latency,
            status_code as status,
            COALESCE(temperature, 0) as temp
        FROM api_logs
        ORDER BY timestamp DESC
        LIMIT ?
    """,
        (limit,),
    )

    rows = cursor.fetchall()
    if rows:
        headers = [
            "Request ID",
            "Time",
            "Model",
            "Provider",
            "P.Tokens",
            "C.Tokens",
            "Total",
            "Latency(ms)",
            "Status",
            "Temp",
        ]
        table_data = []
        for row in rows:
            table_data.append(
                [
                    row["req_id"],
                    row["time"][-8:] if row["time"] else "N/A",  # Just show time part
                    row["model_id"][:30],  # Truncate long model names
                    row["provider"],
                    row["p_tok"] or 0,
                    row["c_tok"] or 0,
                    row["total"] or 0,
                    row["latency"] or 0,
                    row["status"] or "N/A",
                    f"{row['temp']:.1f}" if row["temp"] else "0.0",
                ]
            )
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        print("No logs found.")


def show_statistics(conn: sqlite3.Connection) -> None:
    """Show usage statistics."""
    print("\n Usage Statistics")
    print("=" * 100)

    # Overall stats
    cursor = conn.execute("""
        SELECT
            COUNT(*) as total_requests,
            SUM(CASE WHEN status_code = 200 THEN 1 ELSE 0 END) as successful_requests,
            SUM(total_tokens) as total_tokens_used,
            AVG(latency_ms) as avg_latency,
            MIN(latency_ms) as min_latency,
            MAX(latency_ms) as max_latency
        FROM api_logs
    """)

    stats = cursor.fetchone()
    if stats and stats["total_requests"] > 0:
        print("\n Overall Metrics:")
        print(f"  • Total Requests: {stats['total_requests']}")
        print(
            f"  • Successful: {stats['successful_requests']} ({stats['successful_requests'] * 100 / stats['total_requests']:.1f}%)"
        )
        print(f"  • Total Tokens Used: {stats['total_tokens_used'] or 0:,}")
        print(f"  • Avg Latency: {stats['avg_latency']:.0f}ms")
        print(f"  • Min/Max Latency: {stats['min_latency']:.0f}ms / {stats['max_latency']:.0f}ms")

    # Per model stats
    print("\n Per Model Statistics:")
    cursor = conn.execute("""
        SELECT
            model_id,
            provider,
            COUNT(*) as requests,
            SUM(total_tokens) as tokens,
            AVG(latency_ms) as avg_latency
        FROM api_logs
        GROUP BY model_id, provider
        ORDER BY requests DESC
    """)

    rows = cursor.fetchall()
    if rows:
        headers = ["Model", "Provider", "Requests", "Total Tokens", "Avg Latency(ms)"]
        table_data = []
        for row in rows:
            table_data.append(
                [
                    row["model_id"][:40],  # Truncate long names
                    row["provider"],
                    row["requests"],
                    f"{row['tokens'] or 0:,}",
                    f"{row['avg_latency']:.0f}" if row["avg_latency"] else "N/A",
                ]
            )
        print(tabulate(table_data, headers=headers, tablefmt="grid"))


def show_sample_conversation(conn: sqlite3.Connection) -> None:
    """Show a sample conversation with details."""
    print("\n Sample Conversation Details")
    print("=" * 100)

    # Get a recent successful request
    cursor = conn.execute("""
        SELECT * FROM api_logs
        WHERE status_code = 200 AND response IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT 1
    """)

    row = cursor.fetchone()
    if row:
        print(f"\n Request ID: {row['request_id']}")
        print(f" Timestamp: {row['timestamp']}")
        print(f" Model: {row['model_id']}")
        print(f" Provider: {row['provider']}")
        print(f" Latency: {row['latency_ms']}ms")

        # Parse and show prompt
        if row["prompt"]:
            prompt = json.loads(row["prompt"])
            print("\n Prompt:")
            for msg in prompt:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                print(f"  [{role}]: {content[:100]}...")  # Truncate long messages

        # Parse and show response
        if row["response"]:
            response = json.loads(row["response"])
            if response.get("choices"):
                content = response["choices"][0]["message"]["content"]
                print("\n Response:")
                print(f"  {content[:200]}...")  # Show first 200 chars

        # Show token usage
        print("\n Token Usage:")
        print(f"  • Prompt Tokens: {row['prompt_tokens'] or 0}")
        print(f"  • Completion Tokens: {row['completion_tokens'] or 0}")
        print(f"  • Total Tokens: {row['total_tokens'] or 0}")

        # Show parameters if any
        if row["temperature"] or row["top_p"] or row["max_tokens"]:
            print("\n  Parameters:")
            if row["temperature"]:
                print(f"  • Temperature: {row['temperature']}")
            if row["top_p"]:
                print(f"  • Top-p: {row['top_p']}")
            if row["max_tokens"]:
                print(f"  • Max Tokens: {row['max_tokens']}")
    else:
        print("No successful requests found.")


def show_error_logs(conn: sqlite3.Connection) -> None:
    """Show any errors that occurred."""
    print("\n  Error Logs")
    print("=" * 100)

    cursor = conn.execute("""
        SELECT
            substr(request_id, 5, 8) as req_id,
            datetime(timestamp) as time,
            model_id,
            status_code,
            error
        FROM api_logs
        WHERE error IS NOT NULL OR status_code != 200
        ORDER BY timestamp DESC
        LIMIT 5
    """)

    rows = cursor.fetchall()
    if rows:
        for row in rows:
            print(f" [{row['time']}] Request {row['req_id']} - Model: {row['model_id']}")
            print(f"   Status: {row['status_code']}, Error: {row['error'] or 'N/A'}")
    else:
        print(" No errors found!")


def main() -> None:
    """Main function."""
    print(" OpenRouter Database Log Viewer")
    print("=" * 100)

    try:
        conn = connect_db()

        # Check if database has any logs
        cursor = conn.execute("SELECT COUNT(*) as count FROM api_logs")
        count = cursor.fetchone()["count"]

        if count == 0:
            print("\n  No logs in database yet!")
            print("Make some API requests first using:")
            print("  ./send_test_requests.sh")
            return

        print(f"\n Found {count} log entries in database")

        # Show all sections
        show_recent_logs(conn, limit=10)
        show_statistics(conn)
        show_sample_conversation(conn)
        show_error_logs(conn)

        print("\n" + "=" * 100)
        print(" End of log analysis")
        print("\n Tip: You can query the SQLite database directly:")
        print("  sqlite3 var/db/openrouter_logs.db")
        print("  .tables")
        print("  SELECT * FROM api_logs LIMIT 5;")

        conn.close()

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure the server is running and has processed some requests.")


if __name__ == "__main__":
    main()
