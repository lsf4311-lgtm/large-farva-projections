import requests
import datetime
import os

URL = "https://large-farva-projections-alpha.streamlit.app/"
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keepwarm.log")
TIMEOUT = 60  # seconds — Streamlit cold starts can be slow


def ping():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        response = requests.get(URL, timeout=TIMEOUT)
        status = f"OK ({response.status_code})"
    except requests.exceptions.Timeout:
        status = "TIMEOUT"
    except requests.exceptions.RequestException as e:
        status = f"ERROR: {e}"

    log_line = f"{timestamp} | {status}\n"

    with open(LOG_FILE, "a") as f:
        f.write(log_line)

    print(log_line.strip())


if __name__ == "__main__":
    ping()
