# 20210531 python-mariaDB connection
import mariadb as mariadb
import sys


# Connect to MariaDB Platform
def connectMariaDB():
    try:
        conn = mariadb.connect(
            user="admin",
            password="hist1984!FAB",
            host="mariadb-melchior.cmoghwh13bot.ap-northeast-2.rds.amazonaws.com",
            port=3306,
            database="MELCHIOR"
        )
    except mariadb.Error as e:
        print(f"Error connecting to MariaDB Platform: {e}")
        sys.exit(1)

    return conn


def closeMariaDB(connector=None):
    connector.close()
    return 0
