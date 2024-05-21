import os
import requests


def check_website(url):
    print(f"Checking {url}...")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print(f"✅ {url} is responsive.")
            return True
        else:
            print(f"❌ {url} is not responsive.")
            return False
    except Exception as e:
        print(f"❌ {url} is not responsive. Error: {e}")
        return False


def check_webarena_servers():

    assert check_website(os.environ.get("SHOPPING"))
    assert check_website(os.environ.get("SHOPPING_ADMIN"))
    assert check_website(os.environ.get("REDDIT"))
    assert check_website(os.environ.get("GITLAB"))
    assert check_website(os.environ.get("WIKIPEDIA"))
    assert check_website(os.environ.get("MAP"))
    assert check_website(os.environ.get("HOMEPAGE"))


if __name__ == "__main__":

    check_website(os.environ.get("SHOPPING"))
    check_website(os.environ.get("SHOPPING_ADMIN"))
    check_website(os.environ.get("REDDIT"))
    check_website(os.environ.get("GITLAB"))
    check_website(os.environ.get("WIKIPEDIA"))
    check_website(os.environ.get("MAP"))
    check_website(os.environ.get("HOMEPAGE"))
