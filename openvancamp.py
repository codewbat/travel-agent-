import requests

class Basket:
    def compare(self, from_country: str, to_country: str):
        """Compare cost of living between two countries via OpenVan.camp API."""
        url = "https://openvan.camp/api/vanbasket/compare"
        params = {
            "from": from_country.upper(),
            "to": to_country.upper(),
            "source": "antigravity_travel_planner"
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            res_data = response.json()
            if res_data.get("success"):
                data = res_data.get("data", {})
                diff = data.get("diff_percent", 0)
                cheaper_by = abs(diff) if diff < 0 else 0
                cost_index = data.get("to", {}).get("vanbasket_index", 100)
                spending_power = data.get("budget_100", 100)
                return {
                    'cost_index': cost_index,
                    'percentage_cheaper': cheaper_by,
                    'spending_power': spending_power
                }
        except Exception as e:
            print(f"❌ OpenVan comparison error: {e}")
            
        return {
            'cost_index': 100,
            'percentage_cheaper': 0,
            'spending_power': 100
        }

class OpenVan:
    def __init__(self):
        self.basket = Basket()
