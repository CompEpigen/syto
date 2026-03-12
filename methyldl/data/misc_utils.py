def _merge_comma_separated(str1, str2):
    """Merge two comma-separated strings, keeping unique values."""
    if not str1 and not str2:
        return ""
    items1 = set(str1.split(",")) if str1 else set()
    items2 = set(str2.split(",")) if str2 else set()
    merged = items1 | items2
    merged.discard("")  # Remove empty strings
    return ",".join(sorted(merged))
