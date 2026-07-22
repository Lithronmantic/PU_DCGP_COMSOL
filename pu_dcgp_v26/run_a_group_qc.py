"""Run the frozen A-group outcome-blind QC audit."""

from .a_group_qc import audit_a_group_data, write_a_group_qc


def main() -> None:
    audit, flags = audit_a_group_data()
    summary, table = write_a_group_qc(audit, flags)
    print(summary)
    print(table)


if __name__ == "__main__":
    main()
