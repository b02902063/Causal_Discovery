import requests
import os
from datetime import datetime
import json
import numpy as np
import base64

url = 'https://us-central1-ameai-causal.cloudfunctions.net/discovery'


NO = -1
num_nodes = 21
constraint_matrix = np.full((num_nodes, num_nodes), NO, dtype=np.float32)
constraint_matrix[:, -1] = 0.0
constraint_matrix[8, -1] = NO
constraint_matrix[0, 13] = 0.0
constraint_matrix[14, :] = 0.0
constraint_matrix[:, 16] = 0.0
constraint_matrix[:, 17] = 0.0
constraint_matrix[16, 17] = NO
constraint_matrix[18, :] = 0.0
constraint_matrix[19, :] = 0.0
constraint_matrix[18, 19] = NO

constraint_string = base64.b64encode(constraint_matrix).decode("utf-8")

J = {
    "data": "gs://causal_data/Discovery_data/Rayark/sdorica_player.csv",
    "dtype": "ccccccccCcccccBcccccc",
    "constraint": constraint_string
}

headers = {
    'Authorization': 'Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6IjQ1NmI1MmM4MWUzNmZlYWQyNTkyMzFhNjk0N2UwNDBlMDNlYTEyNjIiLCJ0eXAiOiJKV1QifQ.eyJhdWQiOiIzMjU1NTk0MDU1OS5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsImF6cCI6ImFtZWFpLWNhdXNhbEBhcHBzcG90LmdzZXJ2aWNlYWNjb3VudC5jb20iLCJlbWFpbCI6ImFtZWFpLWNhdXNhbEBhcHBzcG90LmdzZXJ2aWNlYWNjb3VudC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZXhwIjoxNzAzODU2MTAwLCJpYXQiOjE3MDM4NTI1MDAsImlzcyI6Imh0dHBzOi8vYWNjb3VudHMuZ29vZ2xlLmNvbSIsInN1YiI6IjExMjE5NTI3Njg3MzMzODAwMzA2NiJ9.DC6Z_kP1AH5DpCUFIyCDBmJMYq--pMOZAjr0a_pa39vNelUld6Bv0oQ1BHwohop5wT9iuD6kG4XQcAXKLeP2neLCPFK33QXSxr72gz9NuUsItA43sd0oWuhjakQsPPNK3r0uT9_4Abgeevf-2s-Ew2osEwAEIx9v0Qdwq0PML-cnGs2e_tR86a9YwFd0hIl5kKARLtVCJDPUbaQrwV6jrxpEZ3ZRhaAeh5PyQGzAv3mKoxraoml8GmwbDTKXfOmuT-Z9qG6KKH7WY85eAgQSOWADCayQkC26EYJekYrQCCYF6qUSrqLNlJDHsMUt304AdsPrypMeOwZS5OiXCojKSQ',
}

response = requests.post(
    url,
    headers=headers,
    json=J
)

print(response.status_code) 

J = response.json()
print(J)  

result = response.text
print(result)  