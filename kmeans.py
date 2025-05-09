import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import psycopg2
from io import StringIO
import warnings
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Import Axes3D for 3D plotting
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Suppress the SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def connect_to_postgres(dbname, user, password, host, port):
    """
    Connects to a PostgreSQL database.

    Args:
        dbname (str): The name of the database.
        user (str): The username for the database.
        password (str): The password for the database.
        host (str): The host address of the database server.
        port (int): The port number the database server is listening on.

    Returns:
        psycopg2.extensions.connection: A connection object if the connection is successful.
        None: If the connection fails.
    """
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

        # Get the column names from the cursor description.
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
        print("DataFrame is empty. Nothing to insert.")
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
        # Convert pandas dtypes to PostgreSQL data types. This is simplified for demonstration.
        if pd.api.types.is_numeric_dtype(data_type):
            pg_data_type = "NUMERIC"
        elif pd.api.types.is_datetime64_any_dtype(data_type):
            pg_data_type = "TIMESTAMP"
        else:
            pg_data_type = "TEXT"  # Default to TEXT for simplicity

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
    Main function to connect to the database, retrieve data, perform K-means clustering, and visualize the results.
    """
    # Database connection parameters.
    dbname = "crm_service"
    user = "postgres"
    password = "Dha@2023"
    host = "localhost"
    port = 5432
    output_table_name = "customer_segments"  # Table to store clustering results
    output_schema_name = "data_warehouse"  # Target schema name

    # SQL query to retrieve the data. Adjust this query as needed to get your RFM data.
    sql_query = """
    SELECT
        customer_id,
        recency_days,
        frequency,
        monetary_value
    FROM
        data_warehouse.rfm_dataset;  -- Assuming your table is named rfm_dataset
    """

    # Attempt to connect to the database.
    conn = connect_to_postgres(dbname, user, password, host, port)

    # Check if the connection was successful.
    if conn:
        # Execute the query and get the results as a DataFrame.
        df = execute_query_and_fetch_df(conn, sql_query)

        if df is not None and not df.empty:
            print("Data successfully retrieved from the database.")
            
            # Ensure all feature columns are numeric
            features = ['recency_days', 'frequency', 'monetary_value']
            for feature in features:
                df[feature] = pd.to_numeric(df[feature], errors='coerce')
            
            # Drop rows with NaN values that may result from conversion
            df.dropna(subset=features, inplace=True)
            
            # Select the features for clustering
            X = df[features].values  # Convert to numpy array for KMeans

            # Determine the optimal number of clusters (K) using the Elbow Method
            k_range = range(1, 11)
            inertias = []

            for k in k_range:
                # Create a KMeans instance with k clusters
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # Specify n_init as an integer
                # Fit the model to the data
                kmeans.fit(X)
                # Get the inertia
                inertias.append(kmeans.inertia_)

            # Plot the Elbow Method graph
            plt.figure(figsize=(8, 6))
            plt.plot(k_range, inertias, marker='o', linestyle='-', color='b')
            plt.title('Elbow Method for Optimal K')
            plt.xlabel('Number of Clusters (K)')
            plt.ylabel('Inertia')
            plt.xticks(k_range)
            plt.grid(True)
            plt.savefig('elbow_method.png')  # Save figure to file instead of showing
            print("Elbow method plot saved to 'elbow_method.png'")

            # Based on the Elbow Method graph, choose an appropriate value for K
            # Let's assume K=3 for this example (you should choose based on your plot)
            optimal_k = 3

            # Apply K-means clustering with the chosen K
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)  # Specify n_init as an integer
            # Fit the model to the data
            kmeans.fit(X)
            # Get the cluster labels for each data point
            df['cluster'] = kmeans.labels_

            # Calculate validation metrics
            try:
                silhouette_avg = silhouette_score(X, kmeans.labels_)
                calinski_harabasz = calinski_harabasz_score(X, kmeans.labels_)
                davies_bouldin = davies_bouldin_score(X, kmeans.labels_)

                print("\nClustering Validation Metrics:")
                print(f"Silhouette Score: {silhouette_avg:.3f}")
                print(f"Calinski-Harabasz Index: {calinski_harabasz:.3f}")
                print(f"Davies-Bouldin Index: {davies_bouldin:.3f}")
            except Exception as e:
                print(f"Error calculating validation metrics: {e}")

            # Print the cluster assignments
            print("\nCluster Assignments:")
            print(df[['customer_id', 'cluster']].head(10))
            print(f"Total rows: {len(df)}")

            # Calculate and display cluster statistics
            try:
                # Convert to numeric types and calculate means manually
                cluster_means = df.groupby('cluster')[features].apply(lambda x: x.astype(float).mean())
                print("\nCluster Means:")
                print(cluster_means)
                
                # Add cluster descriptions based on means
                cluster_descriptions = []
                for i, means in cluster_means.iterrows():
                    # High-value loyal: highest monetary value and frequency
                    if means['monetary_value'] >= 20000 and means['frequency'] >= 5:
                        description = "High-value loyal customers"
                    # Regular: moderate monetary value and frequency
                    elif means['monetary_value'] >= 10000 or (means['frequency'] >= 3 and means['monetary_value'] >= 5000):
                        description = "Regular customers"
                    # Inactive: low frequency and monetary value
                    else:
                        description = "Inactive customers"
                    cluster_descriptions.append({"cluster": i, "description": description})
                
                # Create a DataFrame for cluster descriptions
                cluster_desc_df = pd.DataFrame(cluster_descriptions)
                print("\nCluster Descriptions:")
                print(cluster_desc_df)
                
                # Merge the original df with cluster descriptions for final output
                df_with_desc = df.merge(cluster_desc_df, on='cluster', how='left')
                
                # Save results to database table
                insert_dataframe_to_table(conn, df_with_desc, output_schema_name, output_table_name)
                
            except Exception as e:
                print(f"Error calculating cluster statistics: {e}")

            # Try to create 3D visualization and save to file
            try:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Create scatter plot with X as numpy arrays
                scatter = ax.scatter(
                    X[:, 0],  # recency_days
                    X[:, 1],  # frequency
                    X[:, 2],  # monetary_value
                    c=kmeans.labels_,
                    cmap='viridis',
                    alpha=0.7
                )
                
                ax.set_title(f'K-means Clustering (K={optimal_k})')
                ax.set_xlabel('Recency (days)')
                ax.set_ylabel('Frequency')
                ax.set_zlabel('Monetary Value')
                
                plt.savefig('kmeans_3d_plot.png')
                print("3D visualization saved to 'kmeans_3d_plot.png'")
            except Exception as e:
                print(f"Error creating 3D visualization: {e}")

        elif df is not None:
            print("Query returned no results.")
        else:
            print("Failed to retrieve data from the database.")

        # Close the connection.
        conn.close()
        print("Connection closed.")
    else:
        print("Failed to connect to the database. Check your credentials and database server status.")


if __name__ == "__main__":
    main()
