from radsimreal.cli import main

if __name__ == "__main__":
    main(["generate", *__import__("sys").argv[1:]])

