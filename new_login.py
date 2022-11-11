import streamlit as st
import pandas as pd
#from multiapp import MultiApp
#from apps import predict
import app
import base64

from PIL import Image


main_bg ="image.jpg"
main_bg_ext = "jpg"

side_bg = "image.jpg"
side_bg_ext = "jpg"

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
   .sidebar .sidebar-content {{
        background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)


import hashlib
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	
	return False
# DB Management
import sqlite3 
#conn = sqlite3.connect('data.db')
conn = sqlite3.connect('data.db', check_same_thread=False)
c = conn.cursor()

# DB  Functions
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data


# def view_all_users():
# 	c.execute('SELECT * FROM userstable')
# 	data = c.fetchall()
# 	return data



def main():
	"""Simple Login App"""

	st.title("Virtual Company")
	

	Menu = ["Employee Login","SignUp"]
	choice = st.sidebar.selectbox("Menu",Menu)

	#if choice == "Home":
		#st.subheader("Home")
		#st.write("Pictorial Represntation Of ")
		#image = Image.open('Picture1.jpg')
		#st.image(image)
	
		
	
		 
			  
		

	if choice == "Employee Login":
		st.subheader("Login Section")
		st.sidebar.write("Please enter your Username and Password")
		username = st.sidebar.text_input("Employee Name")
		password = st.sidebar.text_input("Password",type='password')
		button=st.sidebar.button("Login")
		if(button !="" or st.sidebar.button("Login")):
			# if password == '12345':
			create_usertable()
			hashed_pswd = make_hashes(password)


			result = login_user(username,check_hashes(password,hashed_pswd))

			
			if result:

				st.success("Logged In as {}".format(username))
				PAGES={
					"App1":app,
				}
				
				page = PAGES["App1"]
				page.app()
			
			



				
				
				
	elif choice == "SignUp":
		st.subheader("Create New Account")
		
		new_user = st.text_input("Employee name")
		new_password = st.text_input("Password",type='password')

		if st.button("Signup"):
			create_usertable()
			add_userdata(new_user,make_hashes(new_password))
			st.success("You have successfully created a valid Account")
			st.info("Go to Login Menu to login")
if __name__ == '__main__':
	main()
