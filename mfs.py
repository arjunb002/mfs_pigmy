import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime
import sqlite3
import hashlib
import sys
import warnings
from scipy.optimize import minimize

st.set_page_config(page_title="Bank App", layout="wide")

# --- Top-level page selector ---
page = st.sidebar.radio(
    "Choose Page",
    ["ROI Simulator", "Real Data Dashboard"]
)

if page == "ROI Simulator":
    st.title("🏦 Bank Profitability Dashboard: Traditional vs Your Product")
    # --- Input Section ---
    st.sidebar.header("📥 Input Parameters")
    min_amt = st.sidebar.number_input("Minimum Daily Deposit (₹)", min_value=1, value=10)
    num_customers = st.sidebar.slider("Number of Customers", 100, 10000, 1000, step=100)
    agent_count = st.sidebar.slider("Number of Agents", 1, 100, 10)
    num_branches = st.sidebar.number_input("Number of Branches", min_value=1, value=1)

    # --- Simulation Logic ---
    deposit_days = np.random.randint(30, 366, size=num_customers)
    multipliers = np.random.choice(range(1, 11), size=num_customers)

    # Tiered interest rates
    interest_rates = np.where(
        deposit_days >= 300, 0.04,
        np.where(deposit_days >= 200, 0.035,
                 np.where(deposit_days >= 100, 0.03, 0.025))
    )

    # Financial Calculations
    total_capital = multipliers * min_amt * deposit_days
    interest_expense = total_capital * interest_rates * (deposit_days / 365)
    avg_rate = np.mean(interest_rates)
    total_exp = interest_expense.sum()
    avg_expense = interest_expense.mean()
    total_investment = total_capital.sum()

    # Traditional Bank Costs
    agent_cost = total_investment * 0.03
    conveyance_cost = 1000 * agent_count
    machine_cost = 10000 * agent_count
    total_traditional_cost = total_exp + agent_cost + conveyance_cost + machine_cost

    # Your Product Costs
    total_product_cost = total_exp

    # Savings and ROI
    savings = total_traditional_cost - total_product_cost
    roi = (savings / total_investment) * 100

    # --- Breakdown DataFrame for Average Cost per Customer ---
    avg_cust_convey = conveyance_cost / num_customers
    avg_agent_cost = agent_cost / agent_count
    avg_agent_cust_cost = agent_cost / num_customers
    machine_cost_per_customer = machine_cost / num_customers
    cost_per_customer = avg_expense + avg_agent_cust_cost + avg_cust_convey + machine_cost_per_customer

    breakdown_df = pd.DataFrame([{
        'Average Interest Cost': round(avg_expense, 2),
        'Average Agent-Customer Cost': round(avg_agent_cust_cost, 2),
        'Average Conveyance Cost': round(avg_cust_convey, 2),
        'Average Machine Cost': round(machine_cost_per_customer, 2),
        'Average Cost per Customer': round(cost_per_customer, 2)
    }])

    # --- Display Section ---
    st.subheader("📊 Summary Table")
    data = pd.DataFrame([
        {
            "System": "Traditional",
            "Total Investment (₹)": total_investment,
            "Interest Expense (₹)": total_exp,
            "Operational Cost (₹)": agent_cost + conveyance_cost + machine_cost,
            "Total Cost (₹)": total_traditional_cost,
        },
        {
            "System": "Your Product",
            "Total Investment (₹)": total_investment,
            "Interest Expense (₹)": total_exp,
            "Operational Cost (₹)": 0,
            "Total Cost (₹)": total_product_cost,
        },
    ])
    st.dataframe(
        data.style.format({
            "Total Investment (₹)": "{:.2f}",
            "Interest Expense (₹)": "{:.2f}",
            "Operational Cost (₹)": "{:.2f}",
            "Total Cost (₹)": "{:.2f}",
        })
    )

    st.subheader("🧮 Average Cost per Customer Breakdown")
    st.dataframe(breakdown_df)

    # --- Calculate total simulated points ---
    def calculate_points_banking(amount, deposit_date, withdrawal_date):
        if amount % 10 != 0:
            return 0, 0, 0
        if withdrawal_date <= deposit_date:
            return 0, 0, amount // 10
        earning_days = max(0, (withdrawal_date - deposit_date).days - 1)
        base_points = amount // 10
        total_points = base_points * earning_days
        return total_points, earning_days, base_points

    total_points = 0
    for amt, days in zip(multipliers * min_amt, deposit_days):
        deposit_date = date.today()
        withdrawal_date = deposit_date + timedelta(days=int(days))
        points, _, _ = calculate_points_banking(amt, deposit_date, withdrawal_date)
        total_points += points

    st.metric("Total Simulated Points", total_points)

    # --- Side-by-side comparison for Traditional vs Your Product ---
    if total_points > 0:
        conversion_rate_traditional = total_traditional_cost / total_points
        conversion_rate_product = total_product_cost / total_points
    else:
        conversion_rate_traditional = 0
        conversion_rate_product = 0

    total_payout_traditional = total_points * conversion_rate_traditional
    total_payout_product = total_points * conversion_rate_product

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Traditional System")
        st.metric("Total Expense (₹)", f"{total_traditional_cost:,.2f}")
        st.metric("Conversion Rate per Point (₹)", f"{conversion_rate_traditional:.4f}")
        st.metric("Total Points Payout (₹)", f"{total_payout_traditional:,.2f}")

    with col2:
        st.subheader("Your Product")
        st.metric("Total Expense (₹)", f"{total_product_cost:,.2f}")
        st.metric("Conversion Rate per Point (₹)", f"{conversion_rate_product:.4f}")
        st.metric("Total Points Payout (₹)", f"{total_payout_product:,.2f}")

    # --- Plotting ---
    st.subheader("📉 Cost Comparison")
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = plt.bar(data['System'], data['Total Cost (₹)'], color=['#FF6961', '#77DD77'])
    plt.title("Total Cost Comparison")
    plt.ylabel("Cost in ₹")

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, f"₹{height:,.0f}", ha='center', va='bottom')

    st.pyplot(fig)

    # --- ROI Metric and Total Money Saved ---
    col_roi, col_saved = st.columns(2)
    with col_roi:
        st.metric(label="💹 ROI from Using Your Product", value=f"{roi:.2f}%", delta=f"₹{savings:,.0f} Saved")
    with col_saved:
        total_money_saved = savings * num_branches
        st.metric(label="🏦 Total Money Saved (₹)", value=f"₹{total_money_saved:,.0f}")

    # --- Conclusion ---
    st.markdown("---")
    st.markdown("✅ This dashboard shows how your product eliminates agent, conveyance, and machine costs, leading to significant savings and a healthy ROI for the bank.")

elif page == "Real Data Dashboard":
    # --- Real Data Dashboard code ---
    # (all code from your real data dashboard, including login, deposit, withdraw, analytics, admin panel)
    # --- Helper Functions ---
    def hash_password(password):
        return hashlib.sha256(password.encode()).hexdigest()

    def check_admin(password):
        ADMIN_HASH = '5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8'  # 'password'
        return hash_password(password) == ADMIN_HASH

    def calculate_points(amount, days):
        base = amount // 10
        return base * days * (days + 1) // 2

    # DB Setup
    conn = sqlite3.connect("deposits.db", check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS deposits (
            name TEXT,
            amount INTEGER,
            serial TEXT PRIMARY KEY,
            deposit_date TEXT,
            status TEXT DEFAULT 'active'
        )
    """)
    conn.commit()

    def migrate_remove_primary_key():
        cursor.execute("PRAGMA table_info(deposits)")
        columns = cursor.fetchall()
        serial_is_pk = any(col[1] == 'serial' and col[5] == 1 for col in columns)
        if serial_is_pk:
            cursor.execute("ALTER TABLE deposits RENAME TO deposits_old")
            cursor.execute("""
                CREATE TABLE deposits (
                    name TEXT,
                    amount INTEGER,
                    serial TEXT,
                    deposit_date TEXT,
                    status TEXT DEFAULT 'active'
                )
            """)
            cursor.execute("INSERT INTO deposits (name, amount, serial, deposit_date, status) SELECT name, amount, serial, deposit_date, status FROM deposits_old")
            cursor.execute("DROP TABLE deposits_old")
            conn.commit()

    migrate_remove_primary_key()

    def migrate_add_users_and_timestamp():
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        ''')
        cursor.execute("PRAGMA table_info(deposits)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'user_id' not in columns:
            cursor.execute("ALTER TABLE deposits ADD COLUMN user_id INTEGER")
        if 'timestamp' not in columns:
            cursor.execute("ALTER TABLE deposits ADD COLUMN timestamp TEXT")
        if 'withdrawal_date' not in columns:
            cursor.execute("ALTER TABLE deposits ADD COLUMN withdrawal_date TEXT")
        conn.commit()

    migrate_add_users_and_timestamp()

    def calculate_points_banking(amount, deposit_date, withdrawal_date):
        if amount % 10 != 0:
            return 0, 0, 0
        if withdrawal_date <= deposit_date:
            return 0, 0, amount // 10
        earning_days = max(0, (withdrawal_date - deposit_date).days - 1)
        base_points = amount // 10
        total_points = base_points * earning_days
        return total_points, earning_days, base_points

    def get_dynamic_cap(deposit_days_count):
        if deposit_days_count >= 365:
            return 0.06
        elif deposit_days_count >= 200:
            return 0.04
        elif deposit_days_count >= 100:
            return 0.03
        else:
            return 0.02

    def objective(rate_per_point, points_earned):
        return -points_earned * rate_per_point[0]

    def optimize_conversion_rate(points_earned, principal, cap_rate):
        if principal == 0 or points_earned == 0:
            return 0.0, 0.0
        max_reward = principal * cap_rate
        max_rate_per_point = max_reward / points_earned
        bounds = [(0, max_rate_per_point)]
        x0 = [max_rate_per_point / 2]
        result = minimize(objective, x0, args=(points_earned,), bounds=bounds)
        if result.success:
            rate = result.x[0]
            reward = points_earned * rate
            return round(rate, 6), round(reward, 2)
        else:
            raise Exception("Optimization failed.")

    def signup(username, password):
        try:
            cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, hash_password(password)))
            conn.commit()
            return True, "Signup successful! Please log in."
        except sqlite3.IntegrityError:
            return False, "Username already exists."
        except Exception as e:
            return False, f"Error: {e}"

    def login(username, password):
        cursor.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        if user and user[1] == hash_password(password):
            return True, user[0]
        return False, None

    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = None
    if 'username' not in st.session_state:
        st.session_state['username'] = None
    if 'admin_logged_in' not in st.session_state:
        st.session_state['admin_logged_in'] = False

    def show_login_signup():
        st.sidebar.header("User Login / Signup")
        action = st.sidebar.radio("Action", ["Login", "Signup"])
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if action == "Signup":
            if st.sidebar.button("Signup"):
                success, msg = signup(username.strip(), password)
                if success:
                    st.sidebar.success(msg)
                else:
                    st.sidebar.error(msg)
        else:
            if st.sidebar.button("Login"):
                success, user_id = login(username.strip(), password)
                if success:
                    st.session_state['user_id'] = user_id
                    st.session_state['username'] = username.strip()
                    st.sidebar.success("Logged in successfully!")
                else:
                    st.sidebar.error("Invalid username or password.")
        if st.session_state['user_id']:
            st.sidebar.info(f"Logged in as: {st.session_state['username']}")
            if st.sidebar.button("Logout", key="user_logout"):
                st.session_state['user_id'] = None
                st.session_state['username'] = None
                st.experimental_rerun()

    show_login_signup()

    if not st.session_state['user_id'] and not st.session_state['admin_logged_in']:
        st.stop()

    st.title("💰 Rupee Deposit Credit Point System")

    menu = st.sidebar.radio("Choose Action", ["Deposit", "Withdraw", "Analytics", "Admin Panel"])

    if menu == "Deposit":
        st.header("📝 New Deposit Entry")
        name = st.session_state['username']
        amount = st.number_input("Amount to be deposited (₹10 multiples)", min_value=10, step=10)
        serial = st.text_input("Note serial number")

        if st.button("Submit Deposit"):
            if not serial.strip():
                st.error("Serial number cannot be empty.")
            elif amount % 10 != 0:
                st.error("Amount must be a multiple of ₹10.")
            else:
                deposit_date = date.today().isoformat()
                timestamp = datetime.now().isoformat()
                cursor.execute("SELECT COUNT(*) FROM deposits WHERE serial = ?", (serial.strip(),))
                serial_count = cursor.fetchone()[0]
                if serial_count > 0:
                    status = 'defective'
                    warning_msg = "Duplicate serial detected! Marked as defective or possible fake note, but deposit is allowed."
                else:
                    status = 'active'
                    warning_msg = None
                try:
                    cursor.execute("INSERT INTO deposits (name, amount, serial, deposit_date, status, user_id, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)", (name, amount, serial.strip(), deposit_date, status, st.session_state['user_id'], timestamp))
                    conn.commit()
                    st.success(f"Deposit recorded for {name} on {deposit_date}")
                    if warning_msg:
                        st.warning(warning_msg)
                except Exception as e:
                    st.error(f"Database error: {e}")

    elif menu == "Withdraw":
        st.header("💳 Withdraw & View Points")
        serial = st.text_input("Enter your note serial number")

        if st.button("Check Points"):
            if not serial.strip():
                st.error("Serial number cannot be empty.")
            else:
                cursor.execute("SELECT amount, deposit_date, status, withdrawal_date FROM deposits WHERE user_id = ? AND serial = ? ORDER BY timestamp DESC LIMIT 1", (st.session_state['user_id'], serial.strip()))
                result = cursor.fetchone()

                if result:
                    amount, deposit_date_str, status, withdrawal_date_str = result
                    deposit_date = datetime.strptime(deposit_date_str, "%Y-%m-%d").date()
                    one_year_withdrawal_date = deposit_date + timedelta(days=366)
                    points, earning_days, base_points = calculate_points_banking(amount, deposit_date, one_year_withdrawal_date)
                    cap_rate = get_dynamic_cap(365)
                    rate_per_point, reward = optimize_conversion_rate(points, amount, cap_rate)
                    st.success(f"Total credit points for 1 year: {points}")
                    st.info(f"1 year = {earning_days} point-earning day(s) (₹{base_points} per day)")
                    st.metric("Optimal Rate per Point (₹)", rate_per_point)
                    st.metric("Final Reward after 1 year (₹) provided you invest all 365 days", reward)
                    if status == 'withdrawn' and withdrawal_date_str:
                        withdrawal_date = datetime.strptime(withdrawal_date_str, "%Y-%m-%d").date()
                    else:
                        withdrawal_date = date.today()
                    days_since_deposit = (withdrawal_date - deposit_date).days
                    if days_since_deposit < 366:
                        st.info(f"Withdrawal allowed only after 1 year (365 days) from deposit. {366 - days_since_deposit} day(s) remaining.")
                    else:
                        if status == 'active' and st.button("Withdraw Deposit"):
                            try:
                                today_str = date.today().isoformat()
                                cursor.execute("UPDATE deposits SET status = 'withdrawn', withdrawal_date = ? WHERE user_id = ? AND serial = ? AND status = 'active'", (today_str, st.session_state['user_id'], serial.strip()))
                                conn.commit()
                                st.success("Deposit withdrawn successfully.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error withdrawing deposit: {e}")
                else:
                    st.error("No record found for the given serial number.")

    elif menu == "Analytics":
        st.header("📊 Your Deposit Analytics")
        user_id = st.session_state['user_id']
        df = pd.read_sql_query("SELECT * FROM deposits WHERE user_id = ? ORDER BY timestamp", conn, params=(user_id,))
        if df.empty:
            st.info("No deposits yet.")
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            def calc_banking_points(row):
                if row['status'] == 'withdrawn' and pd.notnull(row['withdrawal_date']):
                    withdrawal_date = pd.to_datetime(row['withdrawal_date']).date()
                else:
                    withdrawal_date = date.today()
                deposit_date = pd.to_datetime(row['deposit_date']).date()
                points, _, _ = calculate_points_banking(row['amount'], deposit_date, withdrawal_date)
                return points
            df['points'] = df.apply(calc_banking_points, axis=1)
            st.subheader("Deposit History")
            st.dataframe(df[['date', 'amount', 'serial', 'status', 'points']])
            total_points = df['points'].sum()
            total_deposit = df['amount'].sum()
            unique_days = df['date'].nunique()
            cap_rate = get_dynamic_cap(unique_days)
            rate_per_point, final_reward = optimize_conversion_rate(total_points, total_deposit, cap_rate)
            st.metric("Total Credit Points", total_points)
            st.metric("Total Deposit Amount (₹)", total_deposit)
            st.metric("Unique Deposit Days", unique_days)
            st.metric("Max Cap Rate Allowed", f"{cap_rate*100:.2f}%")
            st.metric("Optimal Rate per Point (₹)", rate_per_point)
            st.metric("Final Reward (₹)", final_reward)

    elif menu == "Admin Panel":
        st.header("🛠️ Admin Panel: All Deposits & Users")
        admin_pw = st.text_input("Enter admin password", type="password")
        if st.button("Login as Admin"):
            if check_admin(admin_pw):
                st.session_state['admin_logged_in'] = True
            else:
                st.error("Incorrect password.")

        if st.session_state.get('admin_logged_in', False):
            st.subheader("All Users")
            users_df = pd.read_sql_query("SELECT id, username FROM users", conn)
            st.dataframe(users_df)
            st.subheader("All Deposits")
            all_df = pd.read_sql_query("SELECT * FROM deposits", conn)
            st.dataframe(all_df)
            if not all_df.empty:
                all_df['display'] = all_df.apply(lambda row: f"{row['serial']} | {row['name']} | ₹{row['amount']} | {row['deposit_date']} | {row['status']}", axis=1)
                selected_idx = st.selectbox("Select a deposit to delete", all_df.index, format_func=lambda i: all_df.loc[i, 'display'])
                if st.button("Delete Selected Deposit", key="delete_selected_deposit"):
                    row = all_df.loc[selected_idx]
                    cursor.execute("DELETE FROM deposits WHERE serial = ? AND timestamp = ?", (row['serial'], row['timestamp']))
                    conn.commit()
                    st.success(f"Deleted deposit with serial {row['serial']}.")
                    st.rerun()
            else:
                st.info("No deposits to display.")
            st.subheader("Per-User Transactions & Analytics")
            user_options = users_df['username'].tolist()
            selected_user = st.selectbox("Select a user to view their transactions and analytics", user_options)
            if selected_user:
                selected_user_id = users_df[users_df['username'] == selected_user]['id'].values[0]
                user_df = all_df[all_df['user_id'] == selected_user_id].copy()
                st.markdown(f"**User:** {selected_user}")
                if not user_df.empty:
                    user_df['timestamp'] = pd.to_datetime(user_df['timestamp'])
                    user_df['date'] = user_df['timestamp'].dt.date
                    def calc_banking_points(row):
                        if row['status'] == 'withdrawn' and pd.notnull(row['withdrawal_date']):
                            withdrawal_date = pd.to_datetime(row['withdrawal_date']).date()
                        else:
                            withdrawal_date = date.today()
                        deposit_date = pd.to_datetime(row['deposit_date']).date()
                        points, _, _ = calculate_points_banking(row['amount'], deposit_date, withdrawal_date)
                        return points
                    user_df['points'] = user_df.apply(calc_banking_points, axis=1)
                    st.write(user_df[['date', 'amount', 'serial', 'status', 'points']])
                    total_points = user_df['points'].sum()
                    total_deposit = user_df['amount'].sum()
                    unique_days = user_df['date'].nunique()
                    cap_rate = get_dynamic_cap(unique_days)
                    rate_per_point, final_reward = optimize_conversion_rate(total_points, total_deposit, cap_rate)
                    st.metric("Total Credit Points", total_points)
                    st.metric("Total Deposit Amount (₹)", total_deposit)
                    st.metric("Unique Deposit Days", unique_days)
                    st.metric("Max Cap Rate Allowed", f"{cap_rate*100:.2f}%")
                    st.metric("Optimal Rate per Point (₹)", rate_per_point)
                    st.metric("Final Reward (₹)", final_reward)
                else:
                    st.info("No deposits for this user.")
            csv = all_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download All Deposits as CSV",
                data=csv,
                file_name='all_deposits.csv',
                mime='text/csv'
            )
            if st.button("Logout", key="admin_logout"):
                st.session_state['admin_logged_in'] = False
        else:
            st.info("Admin access required to view all deposits and analytics.")
