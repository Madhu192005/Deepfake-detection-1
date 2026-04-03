import traceback

try:
    import main
    print("Success")
except Exception as e:
    with open("error.txt", "w") as f:
        f.write(traceback.format_exc())
