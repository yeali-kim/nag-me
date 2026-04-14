import os
import time
import base64
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By


def build_driver():
    options = Options()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    driver = webdriver.Chrome(options=options)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"}
    )
    return driver


def scrape_google_images(driver, query, folder, max_images):
    os.makedirs(folder, exist_ok=True)

    driver.get(f"https://www.google.com/search?q={query}&tbm=isch&hl=en")
    # need time for captcha if asked again
    time.sleep(10)

    # scroll to load enough thumbnails
    for _ in range(5):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

    thumbnails = [
        img for img in driver.find_elements(By.CSS_SELECTOR, "img")
        if (img.get_attribute("src") or "").startswith("http")
           and img.size.get("width", 0) > 100
    ]
    print(f"{len(thumbnails)} thumbnails for '{query}'")

    count = len([file for file in os.listdir(folder) if file.endswith('.jpg')])
    for thumb in thumbnails:
        if count >= max_images:
            break
        try:
            # open side panel
            driver.execute_script("arguments[0].click();", thumb)
            time.sleep(1.5)

            full_imgs = driver.find_elements(By.CSS_SELECTOR, "img.iPVvYb, img.r48jcc, img.sFlh5c")

            # next image that's not a tracking link
            img_url = next(
                (fi.get_attribute("src") for fi in full_imgs
                 if (fi.get_attribute("src") or "").startswith("http")
                 and "encrypted" not in fi.get_attribute("src")),
                thumb.get_attribute("src")
            )
            if not img_url:
                continue
            # split header if base 64 img
            if img_url.startswith("data:image"):
                _, encoded = img_url.split(",", 1)
                data = base64.b64decode(encoded)
            else:
                resp = requests.get(img_url, timeout=8, headers={"Referer": "https://www.google.com/"})
                if resp.status_code != 200 or len(resp.content) < 1000:
                    continue
                data = resp.content

            with open(f"{folder}/img_{count}.jpg", "wb") as f:
                f.write(data)

            print(f"  Saved {count + 1}/{max_images}")
            count += 1
        except Exception:
            continue

    print(f"Saved {count} images to '{folder} folder'")


if __name__ == "__main__":
    driver = build_driver()

    driver.get("https://www.google.com")
    input("do captcha and press enter")

    queries = [
        ("person biting nails full face", "data/nail_biting", 100),
        ("biting fingernails close up", "data/nail_biting", 200),
        ("person resting chin on hand portrait", "data/safe_action", 100),
        ("person drinking from a mug", "data/safe_action", 200),
        ("person talking on phone", "data/safe_action", 300),
        ("hands in the air saying hi", "data/safe_action", 400),
        ("hands covering full face", "data/safe_action", 500),
    ]

    for query, folder, max_images in queries:
        scrape_google_images(driver, query, folder, max_images)
        time.sleep(3)

    driver.quit()
