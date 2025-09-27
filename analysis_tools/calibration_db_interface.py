import json
import os 
import requests

class CalibrationDBInterface:
    
    def __init__(self, credential_path="./.wctecaldb.credential", calibration_db_url = "https://wcte.caldb.triumf.ca/api/v1/"):
        self.credential_path = credential_path
        self.calibration_db_url = calibration_db_url
        self.get_jwt_token()
        
    def get_jwt_token(self):
        print("Initialise Calibration Database Authentication")
        token_url = self.calibration_db_url+"login"
        # Check if the credential file exists and is readable
        if not os.path.isfile(self.credential_path) or not os.access(self.credential_path, os.R_OK):
            print("Can't find credential path at ",self.credential_path)
            print("See instructions https://wcte.hyperk.ca/documents/calibration-db-apis/v1-api-endpoints-documentation")
            print("Or copy credential file from EOS 'cp /eos/experiment/wcte/calibration_db_credentials/.wctecaldb.credential .' ")
            
            raise FileNotFoundError(f"Credential file not found or not readable: {self.credential_path}")

        # Read credentials from the file (expects shell-style exports or var=val lines)
        credentials = {}
        with open(self.credential_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                credentials[key.strip()] = value.strip().strip('"').strip("'")

        username = credentials.get('WCTECALDB_USERNAME')
        password = credentials.get('WCTECALDB_PASSWORD')

        if not username or not password:
            print("See instructions https://wcte.hyperk.ca/documents/calibration-db-apis/v1-api-endpoints-documentation")
            raise ValueError("WCTECALDB_USERNAME or WCTECALDB_PASSWORD not found in the credentials file.")

        # Make the POST request to get the token
        response = requests.post(
            token_url,
            headers={'Content-Type': 'application/json'},
            data=json.dumps({
                'username': username,
                'password': password
            })
        )
        if response.status_code != 201:
            print(response)
            raise ValueError(f"Unexpected status code {response.status_code}, expected 201.")
        
        # Parse the token from the response
        self.jwt_token = response.json().get('access_token')
        print(self.jwt_token)
        if not self.jwt_token:
            raise ValueError("Failed to retrieve access_token from the response.")

        print("Token successfully retrieved.")
        return self.jwt_token
    
    def print_jwt_token(self):
        print(self.jwt_token)
        
    
    def get_calibration_constants(self, run_number, time, calibration_name, official):
        url = self.calibration_db_url+"calibration_constants/by_validity_period"
        # params = {
        #     "run_number": run_number,
        #     "time": time,
        #     "calibration_name": calibration_name,
        #     "official":official
        # }
        params = {
            "run_number": run_number,
            "time": time,
            "calibration_name": calibration_name,
            "official": official
        }
        headers = {
            "Authorization": f"Bearer {self.jwt_token}"
        }

        response = requests.get(url, params=params, headers=headers)
                
        if response.status_code != 200:
            print(response, response.json())
            raise ValueError(f"Unexpected status code in constant request response {response.status_code}, expected 201. \n"+
                            str(response.json())
                            )
        
        calibration_data = response.json()

        timing_offsets_list = calibration_data[0]['data']
        revision_id = calibration_data[0]['revision_id']
        insert_time = calibration_data[0]['insert_time']
        return timing_offsets_list, revision_id, insert_time
        return