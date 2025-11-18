import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from tkinter import *
from tkinter import messagebox
import sqlite3
import os

# ----------------- GLOBALS & DATA -----------------

# Secret code for owner login
rras_code = "Wyd^H3R"

# Foods list
foods = [
    "Idly", "Dosa", "Vada", "Roti",
    "Meals", "Veg Biryani", "Egg Biryani", "Chicken Biryani",
    "Mutton Biryani", "Ice Cream", "Noodles", "Manchooriya",
    "Orange juice", "Apple Juice", "Pineapple juice", "Banana juice"
]

food_rev = {}
food_perc = {}
for i in foods:
    food_rev[i] = []
    food_perc[i] = [0.0, 0.0]

variables = []       # for customer checkboxes
clr_variables = []   # for clear-item checkboxes

# ----------------- LOAD DATASET & TRAIN MODEL -----------------

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()

    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    # ensure 'not' is kept
    if 'not' in all_stopwords:
        all_stopwords.remove('not')

    review = [ps.stem(word) for word in review if word not in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

# ----------------- SQLITE DB HELPERS -----------------

DB_NAME = 'Restaurant_food_data.db'


def create_db_and_seed():
    """Create the database and table if needed, and insert food items if table is empty."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS item (
            Item_name text,
            No_of_customers text,
            No_of_positive_reviews text,
            No_of_negative_reviews text,
            Positive_percentage text,
            Negative_percentage text
        )
    """)
    conn.commit()

    # Check if table is empty
    c.execute("SELECT COUNT(*) FROM item")
    count = c.fetchone()[0]

    if count == 0:
        for food in foods:
            c.execute("""
                INSERT INTO item
                (Item_name, No_of_customers, No_of_positive_reviews,
                 No_of_negative_reviews, Positive_percentage, Negative_percentage)
                VALUES (:item_name, :no_of_customers, :no_of_positives,
                        :no_of_negatives, :pos_perc, :neg_perc)
            """, {
                'item_name': food,
                'no_of_customers': "0",
                'no_of_positives': "0",
                'no_of_negatives': "0",
                'pos_perc': "0.0%",
                'neg_perc': "0.0%"
            })
        conn.commit()

    conn.close()

# ----------------- CUSTOMER FLOW -----------------


def take_review():
    root2 = Toplevel()
    root2.title(main + "Give Review")

    label = Label(root2, text="RESTAURANT REVIEW ANALYSIS SYSTEM",
                  bd=2, font=('Arial', 47, 'bold', 'underline'))

    req1 = Label(root2, text="Select the item(s) you have taken.....")

    global variables
    variables = []
    chk_btns = []

    # Checkbox for each food item
    for i in range(len(foods)):
        var = IntVar()
        chk = Checkbutton(root2, text=foods[i], variable=var)
        variables.append(var)
        chk_btns.append(chk)

    req2 = Label(root2, text="Give your review below....")
    rev_tf = Entry(root2, width=125, borderwidth=5)
    req3 = Label(root2, text="NOTE : Use 'not' instead of n't.")

    label.grid(row=0, column=0, columnspan=4)
    req1.grid(row=1, column=0, columnspan=4, sticky=W + E)
    req1.config(font=("Helvetica", 30))

    # Arrange checkboxes in 4x4 grid
    for i in range(4):
        for j in range(4):
            c = chk_btns[i * 4 + j]
            c.grid(row=i + 3, column=j, columnspan=1, sticky=W)

    submit_review = Button(
        root2,
        text="Submit Review",
        font=('Arial', 20),
        padx=100,
        pady=20,
        command=lambda: [estimate(rev_tf.get()), root2.destroy()]
    )

    root2.state("zoomed")
    req2.grid(row=7, column=0, columnspan=4, sticky=W + E)
    req2.config(font=("Helvetica", 20))
    rev_tf.grid(row=8, column=1, rowspan=3, columnspan=2, sticky=S)
    req3.grid(row=11, column=1, columnspan=2)
    submit_review.grid(row=12, column=0, columnspan=4)


def estimate(s):
    """Process review, predict sentiment, and update DB for selected items."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    review = re.sub('[^a-zA-Z]', ' ', s)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    if 'not' in all_stopwords:
        all_stopwords.remove('not')

    review = [ps.stem(word) for word in review if word not in set(all_stopwords)]
    review = ' '.join(review)
    X_new = cv.transform([review]).toarray()
    res = classifier.predict(X_new)  # numpy array

    # Flip prediction if 'not' appears
    if "not" in review:
        res[0] = abs(res[0] - 1)

    selected_foods = []
    for i in range(len(foods)):
        if variables[i].get() == 1:
            selected_foods.append(foods[i])

    c.execute("SELECT *, oid FROM item")
    records = c.fetchall()

    for row in records:
        rec = list(row)
        item_name = rec[0]
        oid = rec[-1]
        if item_name in selected_foods:
            n_cust = int(rec[1]) + 1
            n_pos = int(rec[2])
            n_neg = int(rec[3])

            if res[0] == 1:
                n_pos += 1
            else:
                n_neg += 1

            pos_percent = round((n_pos / n_cust) * 100, 1)
            neg_percent = round((n_neg / n_cust) * 100, 1)

            c.execute("""
                UPDATE item SET
                    Item_name = :item_name,
                    No_of_customers = :no_of_customers,
                    No_of_positive_reviews = :no_of_positives,
                    No_of_negative_reviews = :no_of_negatives,
                    Positive_percentage = :pos_perc,
                    Negative_percentage = :neg_perc
                WHERE oid = :Oid
            """, {
                'item_name': item_name,
                'no_of_customers': str(n_cust),
                'no_of_positives': str(n_pos),
                'no_of_negatives': str(n_neg),
                'pos_perc': str(pos_percent) + "%",
                'neg_perc': str(neg_percent) + "%",
                'Oid': oid
            })

    conn.commit()
    conn.close()
    messagebox.showinfo("Thank you", "Your review has been submitted!")


# ----------------- OWNER LOGIN & MENU -----------------


def login():
    root3 = Toplevel()
    root3.title(main + "Owner Verification")

    label = Label(root3, text="RESTAURANT REVIEW ANALYSIS SYSTEM",
                  bd=2, font=('Arial', 47, 'bold', 'underline'))

    label2 = Label(root3, text="VERIFY OWNERSHIP",
                   bd=1, font=('Helvetica', 30, 'bold', 'underline'))

    label3 = Label(
        root3,
        text="To verify your ownership, please enter your restaurant's private RRAS code...",
        bd=1,
        font=('Helvetica', 20, 'bold')
    )

    ent = Entry(root3, show="*", borderwidth=2)

    submit_code = Button(
        root3,
        text="Submit",
        font=('Arial', 20),
        padx=80,
        pady=20,
        command=lambda: [view_details(ent.get()), root3.destroy()]
    )

    root3.state("zoomed")
    label.grid(row=0, column=0, columnspan=3)
    label2.grid(row=1, column=0, sticky=W + E, columnspan=3)
    label3.grid(row=2, column=0, sticky=W, columnspan=3)
    ent.grid(row=3, column=1, columnspan=1)
    submit_code.grid(row=4, column=1, columnspan=1)


def popup():
    messagebox.showerror("Error", "Incorrect code!")


def view_details(s):
    if s != rras_code:
        popup()
        return

    root4 = Toplevel()
    root4.title(main + "Owner Panel")

    label = Label(root4, text="RESTAURANT REVIEW ANALYSIS SYSTEM",
                  bd=2, font=('Arial', 47, 'bold', 'underline'))

    sug1 = Label(
        root4,
        text="Click the below button if you want to view the data from your database...."
    )

    acc_btn = Button(
        root4,
        text="View Data",
        font=('Arial', 20),
        padx=100,
        pady=20,
        command=access_data
    )

    sug2 = Label(
        root4,
        text="Click the below button if you want to clear specific item data..."
    )

    itemclr_btn = Button(
        root4,
        text="Clear Item Data",
        font=('Arial', 20),
        padx=100,
        pady=20,
        command=clr_itemdata
    )

    sug3 = Label(
        root4,
        text="Click the below button if you want to clear all item data..."
    )

    allclr_btn = Button(
        root4,
        text="Clear All Data",
        font=('Arial', 20),
        padx=100,
        pady=20,
        command=clr_alldata
    )

    exit_btn = Button(root4, text="Exit", command=root4.destroy)

    root4.state("zoomed")
    label.grid(row=0, column=0)
    sug1.grid(row=1, column=0)
    sug1.config(font=("Helvetica", 30))
    acc_btn.grid(row=2, column=0)
    sug2.grid(row=3, column=0)
    sug2.config(font=("Helvetica", 30))
    itemclr_btn.grid(row=4, column=0)
    sug3.grid(row=5, column=0)
    sug3.config(font=("Helvetica", 30))
    allclr_btn.grid(row=6, column=0)
    exit_btn.grid(row=9, column=0, sticky=S)


def access_data():
    root5 = Toplevel()
    root5.title(main + "Restaurant Database")

    label = Label(root5, text="RESTAURANT REVIEW ANALYSIS SYSTEM",
                  bd=2, font=('Arial', 47, 'bold', 'underline'))

    title1 = Label(root5, text="S.NO", font=('Arial', 10, 'bold', 'underline'))
    title2 = Label(root5, text="FOOD ITEM", font=('Arial', 10, 'bold', 'underline'))
    title3 = Label(root5, text="NO.OF CUSTOMERS", font=('Arial', 10, 'bold', 'underline'))
    title4 = Label(root5, text="NO.OF POSITIVE REVIEWS", font=('Arial', 10, 'bold', 'underline'))
    title5 = Label(root5, text="NO.OF NEGATIVE REVIEWS", font=('Arial', 10, 'bold', 'underline'))
    title6 = Label(root5, text="POSITIVE RATE", font=('Arial', 10, 'bold', 'underline'))
    title7 = Label(root5, text="NEGATIVE RATE", font=('Arial', 10, 'bold', 'underline'))

    label.grid(row=0, column=0, columnspan=7)
    title1.grid(row=1, column=0)
    title2.grid(row=1, column=1)
    title3.grid(row=1, column=2)
    title4.grid(row=1, column=3)
    title5.grid(row=1, column=4)
    title6.grid(row=1, column=5)
    title7.grid(row=1, column=6)

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("SELECT *, oid from item")
    records = c.fetchall()
    pos_rates = []

    for record in records:
        record = list(record)
        pos_rates.append(float(record[-3][:-1]))  # strip %

    max_pos = max(pos_rates) if pos_rates else 0
    min_pos = min(pos_rates) if pos_rates else 0

    for i in range(len(records)):
        rec_list = list(records[i])
        pos_str = rec_list[-3]
        rate_val = float(pos_str[:-1]) if pos_str.endswith("%") else 0.0

        if rate_val == max_pos:
            color = "green"
        elif rate_val == min_pos:
            color = "red"
        else:
            color = "black"

        rec_lab = [Label(root5, text=str(rec_list[-1]), fg=color)]
        for item in rec_list[:-1]:
            lab = Label(root5, text=item, fg=color)
            rec_lab.append(lab)

        for j in range(len(rec_lab)):
            rec_lab[j].grid(row=i + 2, column=j)

    exit_btn = Button(root5, text="Exit", command=root5.destroy)
    exit_btn.grid(row=len(records) + 5, column=3)

    conn.close()
    root5.state("zoomed")


def clr_itemdata():
    root6 = Toplevel()
    root6.title(main + "Clear Item Data")

    label = Label(root6, text="RESTAURANT REVIEW ANALYSIS SYSTEM",
                  bd=2, font=('Arial', 47, 'bold', 'underline'))

    req1 = Label(root6, text="Pick the items to clear their corresponding item data....")

    global clr_variables
    clr_variables = []
    chk_list = []

    for i in range(len(foods)):
        var = IntVar()
        chk = Checkbutton(root6, text=foods[i], variable=var)
        clr_variables.append(var)
        chk_list.append(chk)

    label.grid(row=0, column=0, columnspan=4)
    req1.grid(row=1, column=0, columnspan=4, sticky=W + E)
    req1.config(font=("Helvetica", 30))

    for i in range(4):
        for j in range(4):
            c = chk_list[i * 4 + j]
            c.grid(row=i + 3, column=j, columnspan=1, sticky=W)

    clr_item = Button(
        root6,
        text="Clear",
        font=('Arial', 20),
        padx=100,
        pady=20,
        command=lambda: [clr_data(), root6.destroy()]
    )

    clr_item.grid(row=8, column=0, columnspan=4)
    root6.state("zoomed")


def clr_data():
    """Clear data only for selected items."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    for i, var in enumerate(clr_variables):
        if var.get() == 1:
            c.execute("""
                UPDATE item SET
                    Item_name = :item_name,
                    No_of_customers = :no_of_customers,
                    No_of_positive_reviews = :no_of_positives,
                    No_of_negative_reviews = :no_of_negatives,
                    Positive_percentage = :pos_perc,
                    Negative_percentage = :neg_perc
                WHERE oid = :Oid
            """, {
                'item_name': foods[i],
                'no_of_customers': "0",
                'no_of_positives': "0",
                'no_of_negatives': "0",
                'pos_perc': "0.0%",
                'neg_perc': "0.0%",
                'Oid': i + 1
            })
    conn.commit()
    conn.close()
    messagebox.showinfo("Done", "Selected item data cleared!")


def clr_alldata():
    confirm = messagebox.askquestion(
        "Confirmation", "Are you sure to delete all data?"
    )

    if confirm == "yes":
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        for i in range(len(foods)):
            c.execute("""
                UPDATE item SET
                    Item_name = :item_name,
                    No_of_customers = :no_of_customers,
                    No_of_positive_reviews = :no_of_positives,
                    No_of_negative_reviews = :no_of_negatives,
                    Positive_percentage = :pos_perc,
                    Negative_percentage = :neg_perc
                WHERE oid = :Oid
            """, {
                'item_name': foods[i],
                'no_of_customers': "0",
                'no_of_positives': "0",
                'no_of_negatives': "0",
                'pos_perc': "0.0%",
                'neg_perc': "0.0%",
                'Oid': i + 1
            })
        conn.commit()
        conn.close()
        messagebox.showinfo("Done", "All data cleared!")


# ----------------- MAIN WINDOW -----------------

def main_window():
    global root1, main
    root1 = Tk()
    main = "Restaurant Review Analysis System/"

    root1.title(main + "Welcome Page")

    label = Label(root1, text="RESTAURANT REVIEW ANALYSIS SYSTEM",
                  bd=2, font=('Arial', 47, 'bold', 'underline'))

    ques = Label(root1, text="Are you a Customer or Owner ???")
    cust = Button(root1, text="Customer", font=('Arial', 20),
                  padx=80, pady=20, command=take_review)
    owner = Button(root1, text="Owner", font=('Arial', 20),
                   padx=100, pady=20, command=login)

    root1.state("zoomed")
    label.grid(row=0, column=0)
    ques.grid(row=1, column=0, sticky=W + E)
    ques.config(font=("Helvetica", 30))
    cust.grid(row=2, column=0)
    owner.grid(row=3, column=0)

    root1.mainloop()


if __name__ == "__main__":
    create_db_and_seed()
    main_window()
