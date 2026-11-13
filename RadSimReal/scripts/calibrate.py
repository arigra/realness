from radsimreal.cli import main

if __name__ == "__main__":
    main(["calibrate", *__import__("sys").argv[1:]])

