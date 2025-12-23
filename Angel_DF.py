from SmartApi.smartConnect import SmartConnect
import pyotp
import pandas as pd
import requests
import pandas as pd
from datetime import datetime
from datetime import datetime, timedelta
import pandas_ta as ta

url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
instruments = pd.DataFrame(requests.get(url).json())

