import subprocess, sys, os

result = subprocess.run(
    [sys.executable, "src/main.py"],
    capture_output=True,
    text=True,
    cwd=r"C:\Users\Hp\Desktop\boq-classification\boq-classification"
)

with open("captured_output.txt", "w") as f:
    f.write("=== STDOUT ===\n")
    f.write(result.stdout)
    f.write("\n=== STDERR ===\n")
    f.write(result.stderr)
    f.write(f"\n=== RETURN CODE: {result.returncode} ===\n")

print("Done. Check captured_output.txt")
