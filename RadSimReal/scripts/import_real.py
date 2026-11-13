from radsimreal.cli import main

if __name__ == "__main__":
    main(["import-real", *__import__("sys").argv[1:]])

