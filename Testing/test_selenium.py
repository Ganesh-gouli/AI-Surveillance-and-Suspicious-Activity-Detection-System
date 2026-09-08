from selenium import webdriver


def test_open_browser():
    driver = webdriver.Chrome()

    driver.get("https://www.facebook.com")

    print("Page title:", driver.title)

    assert "facebook" in driver.title.lower()

    driver.quit()