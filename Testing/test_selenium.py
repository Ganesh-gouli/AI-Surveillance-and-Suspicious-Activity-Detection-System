from selenium import webdriver
from selenium.webdriver.common.by import By


def test_application_loads():
    driver = webdriver.Chrome()

    driver.get("http://localhost:5173/")

    # Verify that the page has loaded
    assert driver.find_element(By.TAG_NAME, "body").is_displayed()

    print("Application loaded successfully")

    driver.quit()