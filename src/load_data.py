import os
from configparser import ConfigParser
import psycopg2
import pandas as pd
from sshtunnel import SSHTunnelForwarder


class LoadData:
    def __init__(self, config_name, db_name):
        self._config = None
        self.init_config(config_name)
        self._db_username = self._config['psql']['username']
        self._db_password = self._config['psql']['password']
        self._db_name = self._config[db_name]['db_name']
        self._schema_name = self._config[db_name]['schema']
        self._sqlhost = self._config['psql']['host']
        self._sqlport = self._config.getint('psql', 'port')
        self._con = None
        self._ssh_username = self._config['ssh']['user']
        self._ssh_password = self._config['ssh']['password']
        self._server_host = self._config['ssh']['host']
        self._ssh_port = self._config.getint('ssh', 'port')
        self.default_location = os.path.abspath(os.curdir)

    def init_config(self, config_name):
        """

        :param config_name:
        :return:
        """
        if self._config is not None:
            return
        self._config = ConfigParser()
        self._config.read(config_name)

    @staticmethod
    def single_insert(conn, insert_req):
        cursor = conn.cursor()
        try:
            cursor.execute(insert_req)
            conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error: %s" % error)
            conn.rollback()
            cursor.close()
            return 1
        cursor.close()

    def query_db(self, query, params=None):
        """
            query the database

            :param dict params: the params to be add to the query in the form of {param_name: value}
            :param str query: The sql query in a string format
            :return: Data-frame with the result of the query
            """
        SSHTunnelForwarder.daemon_forward_servers = True  # fix problems with python >= 3.7
        with SSHTunnelForwarder(
                (self._server_host, self._ssh_port),
                ssh_username=self._ssh_username,
                ssh_password=self._ssh_password,
                remote_bind_address=(self._sqlhost, self._sqlport)
        ) as server:
            server.start()
            _con = psycopg2.connect(dbname=self._db_name,
                                    user=self._db_username,
                                    password=self._db_password,
                                    host=self._sqlhost,
                                    port=server.local_bind_port)
            tr = pd.read_sql(query, _con, params=params)
        return tr

    def query_and_save(self, query, params=None, location=os.curdir, file_name='result'):
        """
        this method queries the database and saves it to csv instead of returning the actual pandas data-frame

        :param dict params: the params to add to the query in the form of {param_name: value}
        :param str query: The sql query in a string format
        :param str location: path to where to save the query
        :param file_name: the file name of the output csv
        :return: None
        """
        try:
            assert not file_name.endswith('.csv')
        except AssertionError:
            raise SyntaxError("The file name should not contain 'csv' ending ")
        df = self.query_db(query, params)
        df.to_csv(os.path.join(location, f"{file_name}.csv"))
