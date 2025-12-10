import joblib


def compare_feature_files(path1, path2):
    print(f"--- Comparing ---\n1: {path1}\n2: {path2}\n")

    try:
        f1 = joblib.load(path1)
        f2 = joblib.load(path2)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # 1. Check Exact Match
    if f1 == f2:
        print(
            "✅ SUCCESS: Both feature lists are IDENTICAL (same features, same order)."
        )
        return

    # 2. Check Length
    print(f"Length: File 1 has {len(f1)}, File 2 has {len(f2)}")

    # 3. Check Content (Ignoring Order)
    set1 = set(f1)
    set2 = set(f2)

    if set1 == set2:
        print("⚠️ WARNING: Same features, but DIFFERENT ORDER.")
        # Optional: Print first mismatch
        for i, (a, b) in enumerate(zip(f1, f2)):
            if a != b:
                print(f"   First mismatch at index {i}: '{a}' vs '{b}'")
                break
    else:
        print("❌ ERROR: Different features found.")

        # What is in 1 but not 2?
        missing_in_2 = set1 - set2
        if missing_in_2:
            print(f"\nFeatures in File 1 but MISSING in File 2 ({len(missing_in_2)}):")
            print(missing_in_2)

        # What is in 2 but not 1?
        missing_in_1 = set2 - set1
        if missing_in_1:
            print(f"\nFeatures in File 2 but MISSING in File 1 ({len(missing_in_1)}):")
            print(missing_in_1)


# --- RUN COMPARISON ---
# Replace these paths with your actual file paths
path_a = "data/feature_order/minutes_feature_order.pkl"
path_b = "data/feature_order/points_feature_order.pkl"

compare_feature_files(path_a, path_b)
