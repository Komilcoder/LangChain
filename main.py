from dotenv import load_dotenv
import os
load_dotenv()


def main():
    print("hello from langchain: ")
    print(os.environ.get("OPENAI_API_KEY"))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/



