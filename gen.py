import google.generativeai as genai

# Configure the API key
genai.configure(api_key="AIzaSyBHoiIctFOUYCgKeWRSUa43pmRtiO90Qzs")

# Initialize the model
model = genai.GenerativeModel("gemini-2.0-flash")  # Make sure the model name is correct

# Generate content
response = model.generate_content("what is supervised learning in five line")

# Print the response text
print(response.text)



#AIzaSyA9CKy2LRifJTakxlhztC-OGSR6XRFKEJo
