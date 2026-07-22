"""Run B-group outcome-blind quality control."""

from .b_group_qc import audit_b_group_data, write_b_group_qc


def main() -> None:
    audit, flags = audit_b_group_data()
    paths = write_b_group_qc(audit, flags)
    print(audit)
    print(*(str(path.resolve()) for path in paths), sep="\n")


if __name__ == "__main__":
    main()
