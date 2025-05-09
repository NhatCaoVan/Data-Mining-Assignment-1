import psycopg2
import pandas as pd
from io import StringIO  # Import StringIO
import time


def connect_to_postgres(dbname, user, password, host, port):
    conn = None
    try:
        # Attempt to establish a connection to the PostgreSQL database.
        conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        print("Successfully connected to the database!")
        return conn
    except Exception as e:
        # Handle any exceptions that occur during the connection attempt.
        print(f"Error connecting to the database: {e}")
        return None



def execute_query_and_fetch_df(conn, query):
    """
    Executes a SQL query and returns the result as a Pandas DataFrame.

    Args:
        conn (psycopg2.extensions.connection): A valid PostgreSQL database connection.
        query (str): The SQL query to execute.

    Returns:
        pandas.DataFrame: A DataFrame containing the query results.
                        Returns an empty DataFrame if the query returns no rows
                        or if an error occurs.
        None: Returns None if there is an error.
    """
    try:
        # Create a cursor object to interact with the database.
        cur = conn.cursor()

        # Execute the SQL query.
        cur.execute(query)

        # Fetch all the rows returned by the query.
        rows = cur.fetchall()

        # Get the column names from the cursor description.  This is important
        # for creating the DataFrame.
        column_names = [desc[0] for desc in cur.description]

        # Create a Pandas DataFrame from the fetched rows and column names.
        df = pd.DataFrame(rows, columns=column_names)
        return df

    except Exception as e:
        print(f"Error executing query or fetching data: {e}")
        return None
    finally:
        # Ensure the cursor is closed, even if an error occurs.
        if 'cur' in locals() and cur:
            cur.close()



def insert_dataframe_to_table(conn, df, schema_name, table_name):
    """
    Inserts the data from a Pandas DataFrame into a PostgreSQL table within a specified schema.
    Creates the table if it does not exist.

    Args:
        conn (psycopg2.extensions.connection): A valid PostgreSQL database connection.
        df (pandas.DataFrame): The DataFrame to insert into the table.
        schema_name (str): The name of the schema where the table resides.
        table_name (str): The name of the table to insert into.
    """
    if df.empty:
        print("DataFrame is empty.  Nothing to insert.")
        return

    # 1. Create a cursor
    cur = conn.cursor()
    
    # 2. Create the schema if it doesn't exist
    try:
        schema_sql = f"CREATE SCHEMA IF NOT EXISTS {schema_name}"
        cur.execute(schema_sql)
        conn.commit()
        print(f"Schema {schema_name} created (if it did not exist).")
    except Exception as e:
        print(f"Error creating schema: {e}")
        conn.rollback()
        cur.close()
        return

    # 3. Create the table if it does not exist.
    # Construct the CREATE TABLE statement based on the DataFrame's columns and their data types.
    create_table_sql = f"CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} ("
    for i, (column_name, data_type) in enumerate(df.dtypes.items()):
        # Convert pandas dtypes to PostgreSQL data types.  This is simplified for demonstration.
        if pd.api.types.is_numeric_dtype(data_type):
            pg_data_type = "NUMERIC"
        elif pd.api.types.is_datetime64_any_dtype(data_type):
            pg_data_type = "TIMESTAMP"
        else:
            pg_data_type = "NUMERIC"  # Default to TEXT for simplicity

        create_table_sql += f"\"{column_name}\" {pg_data_type}"
        if i < len(df.columns) - 1:
            create_table_sql += ", "  # Add comma for next column
    create_table_sql += ")"

    try:
        cur.execute(create_table_sql)
        conn.commit()
        print(f"Table {schema_name}.{table_name} created (if it did not exist).")
    except Exception as e:
        print(f"Error creating table: {e}")
        conn.rollback()
        cur.close()  # Close the cursor here to avoid using it in the COPY operation if table creation failed.
        return  # Exit the function if table creation failed

    # 4. Create a string buffer for the data
    s_buf = StringIO()
    df.to_csv(s_buf, index=False, header=True)  # Write the dataframe to the buffer
    s_buf.seek(0)  # Go back to the start of the buffer

    # 5. Prepare columns with quotes to handle special characters
    columns = ', '.join([f'"{col}"' for col in df.columns])
    
    try:
        # Use COPY command with proper column formatting
        cur.copy_expert(f"COPY {schema_name}.{table_name} ({columns}) FROM STDIN WITH CSV HEADER", s_buf)
        conn.commit()
        print(f"Successfully inserted {len(df)} rows into table {schema_name}.{table_name}")
    except Exception as e:
        print(f"Error inserting data into table {schema_name}.{table_name}: {e}")
        conn.rollback()  # Rollback the transaction on error.
    finally:
        cur.close()



def main():
    """
    Main function to demonstrate connecting to a PostgreSQL database,
    executing a query, and inserting the results into a new table in a specified schema.
    """
    # Database connection parameters.
    dbname = "crm_service"
    user = "postgres"
    password = "Dha@2023"
    host = "localhost"
    port = 5432
    output_table_name = "rfm_dataset"  # Replace with your desired table name
    output_schema_name = "data_warehouse"  # Replace with the target schema name

    # The SQL query to execute.
    sql_query = """
    WITH analysis_date_cte AS (
        -- Đặt ngày phân tích. Thông thường là ngày sau ngày giao dịch cuối cùng trong dữ liệu,
        -- hoặc ngày hiện tại nếu dữ liệu được cập nhật liên tục.
        -- Dựa trên thông tin bạn cung cấp "Current time is Thursday, May 8, 2025",
        -- chúng ta sẽ lấy ngày 2025-05-09 00:00:00'::TIMESTAMP AS analysis_date
        SELECT '2025-01-01 00:00:00'::TIMESTAMP AS analysis_date
    ),
    customer_transactions AS (
        -- Tính toán giá trị cho mỗi dòng giao dịch (mỗi sản phẩm trong một lần mua)
        SELECT
            p.customer_id,
            p.purchase_time,
            (p.quantity * pr.price) AS transaction_value
        FROM
            pos p
        JOIN
            product pr ON p.product_code = pr.product_code
        WHERE
            p.customer_id IS NOT NULL -- Chỉ xem xét các giao dịch có customer_id
    ),
    customer_rfm_raw AS (
        -- Tính toán các giá trị thô cho R (dưới dạng ngày mua cuối), F, M cho mỗi khách hàng
        SELECT
            ct.customer_id,
            MAX(ct.purchase_time) AS last_purchase_date,
            -- Đếm số lần giao dịch riêng biệt (dựa trên thời điểm mua hàng)
            COUNT(DISTINCT ct.purchase_time) AS frequency,
            SUM(ct.transaction_value) AS monetary_value
        FROM
            customer_transactions ct
        GROUP BY
            ct.customer_id
    )
    -- Tính toán Recency cuối cùng và kết hợp R, F, M
    SELECT
        rfm.customer_id,
        -- Tính Recency bằng số ngày từ ngày phân tích đến ngày mua cuối cùng
        EXTRACT(DAY FROM (ad.analysis_date - rfm.last_purchase_date)) AS recency_days,
        rfm.frequency,
        rfm.monetary_value
    FROM
        customer_rfm_raw rfm,
        analysis_date_cte ad
    ORDER BY
        rfm.customer_id;
    """

    # Attempt to connect to the database.
    conn = connect_to_postgres(dbname, user, password, host, port)

    # Check if the connection was successful.
    if conn:
        # Execute the query and get the results as a DataFrame.
        df = execute_query_and_fetch_df(conn, sql_query)

        if df is not None and not df.empty:
            # Insert the DataFrame into the new table.
            insert_dataframe_to_table(conn, df, output_schema_name, output_table_name)
        elif df is not None: #check if it is an empty dataframe
            print("Query returned no results.  Nothing to insert.")
        else:
            print("Failed to retrieve data from the database.")

        # Close the connection.
        conn.close()
        print("Connection closed.")
    else:
        print("Failed to connect to the database.  Check your credentials and database server status.")



if __name__ == "__main__":
    main()
