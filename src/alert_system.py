import os
import sys
import time
import pandas as pd

# --- 0. PATH FIX ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import TWILIO_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, TWILIO_TO_NUMBER

# Twilio Import
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

# Selenium Imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

class AlertSystem:
    def __init__(self):
        try:
            self.client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
            print("✅ Twilio Client Initialized.")
        except Exception as e:
            print(f"❌ Twilio Init Failed: {e}")
            self.client = None
        
    def send_whatsapp_message(self, message_body):
        """Sends a WhatsApp message using Twilio."""
        if not self.client:
            print("⚠️ Twilio client not ready. Alert skipped.")
            return
        try:
            from_whatsapp = f"whatsapp:{TWILIO_FROM_NUMBER}"
            to_whatsapp = f"whatsapp:{TWILIO_TO_NUMBER}"
            
            message = self.client.messages.create(
                body=message_body,
                from_=from_whatsapp,
                to=to_whatsapp
            )
            print(f"✅ WhatsApp Sent! SID: {message.sid}")
            
        except TwilioRestException as e:
            if e.code == 63007:
                print(f"❌ WhatsApp Setup Error (63007): {e.msg}\n   Fix: Enable WhatsApp on From number or use Sandbox.")
            else:
                print(f"❌ WhatsApp Send Error: {e}")
        except Exception as e:
            print(f"❌ WhatsApp Send Error: {e}")

    def fetch_major_economic_events(self):
        """
        Scrapes ForexFactory for high-impact USD events.
        """
        CALENDAR_URL = "https://www.forexfactory.com/calendar?day=today"
        print(f"\n📅 Checking Economic Calendar: {CALENDAR_URL}")
        
        events = []
        
        # Robust Chrome Options
        options = webdriver.ChromeOptions()
        options.add_argument('--headless=new')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        driver = None
        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            driver.set_page_load_timeout(30)
            driver.get(CALENDAR_URL)
            
            # Wait explicitly for the calendar table
            wait = WebDriverWait(driver, 15)
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "calendar__table")))
            
            # Find all event rows
            rows = driver.find_elements(By.CSS_SELECTOR, "tr.calendar__row")
            print(f"🔍 Inspecting {len(rows)} calendar rows...")
            
            for row in rows:
                try:
                    # 1. Filter for High Impact (Red Folder)
                    # We look for the span class that denotes high impact
                    try:
                        impact_ele = row.find_element(By.CLASS_NAME, 'calendar__impact')
                        if 'icon--ff-impact-red' not in impact_ele.find_element(By.TAG_NAME, 'span').get_attribute('class'):
                            continue # Skip non-red events
                    except NoSuchElementException:
                        continue # Skip spacer rows

                    # 2. Filter for USD Currency
                    currency = row.find_element(By.CLASS_NAME, 'calendar__currency').text.strip()
                    if 'USD' not in currency:
                        continue

                    # 3. Extract Data
                    event_name = row.find_element(By.CLASS_NAME, 'calendar__event-title').text.strip()
                    event_time = row.find_element(By.CLASS_NAME, 'calendar__time').text.strip()
                    
                    # Extract Data Columns (Previous, Forecast, Actual)
                    previous = row.find_element(By.CLASS_NAME, 'calendar__previous').text.strip()
                    forecast = row.find_element(By.CLASS_NAME, 'calendar__forecast').text.strip()
                    actual = row.find_element(By.CLASS_NAME, 'calendar__actual').text.strip()
                    
                    # 4. Construct Message based on Data Availability
                    # If 'actual' is empty or just a placeholder, the event is UPCOMING
                    if not actual or actual == '':
                        status = "UPCOMING"
                        msg = (f"📅 *UPCOMING EVENT ALERT*\n"
                               f"🇺🇸 *{event_name}*\n"
                               f"⏰ Time: {event_time}\n"
                               f"📉 Previous: {previous}\n"
                               f"📊 Forecast: {forecast}")
                    else:
                        status = "RELEASED"
                        msg = (f"🚨 *EVENT DATA RELEASED*\n"
                               f"🇺🇸 *{event_name}*\n"
                               f"📉 Previous: {previous}\n"
                               f"📊 Forecast: {forecast}\n"
                               f"💥 *ACTUAL: {actual}*")
                    
                    print(f"   -> Found {status}: {event_name} (Act: {actual}, Fcst: {forecast})")
                    events.append(msg)
                            
                except NoSuchElementException:
                    continue # Safely skip rows that don't match standard structure
                except Exception as e:
                    print(f"Error parsing row: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ Calendar Scrape Failed: {e}")
        finally:
            if driver:
                driver.quit()
        
        return events

# --- Main Runner ---
def run_alert_system():
    alerts = AlertSystem()
    
    print("--- Starting Economic Calendar Scan ---")
    high_impact_events = alerts.fetch_major_economic_events()
    
    if not high_impact_events:
        print("ℹ️ No high-impact USD events found right now.")
    else:
        print(f"✅ Sending {len(high_impact_events)} alerts...")
        for event_msg in high_impact_events:
            alerts.send_whatsapp_message(event_msg)
    
    print("--- Scan Complete ---")

if __name__ == '__main__':
    run_alert_system()