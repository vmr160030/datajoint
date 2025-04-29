import wget
import os
import subprocess
import datajoint as dj

def create_database(home_dir, db_dir, name):
    # check if already exists
    if os.path.isdir(db_dir + '/' + name):
        print('Database path already exists')
        return
    # add directory titled name under home
    os.mkdir(db_dir + '/' + name)
    wget.download('https://raw.githubusercontent.com/datajoint/mysql-docker/master/docker-compose.yaml',
                 db_dir + '/' + name + '/docker-compose.yaml')
    os.chdir(home_dir)

def start_database(home_dir, db_dir, name):
    if not os.path.isdir(db_dir + '/' + name):
        print("create database first!")
        return
    os.chdir(db_dir + '/' + name)
    subprocess.run(['docker', 'compose', 'up', '-d'])
    os.chdir(home_dir)

def stop_database(home_dir, db_dir, name, conn: dj.Connection):
    if conn and conn.is_connected:
        conn.close()
        print('Disconnected')
    if not os.path.isdir(db_dir + '/' + name):
        print('Invalid database name')
        return
    os.chdir(db_dir + '/' + name)
    subprocess.run(['docker', 'compose', 'down'])
    os.chdir(home_dir)

def delete_database(home_dir, db_dir, name, conn: dj.Connection):
    if not os.path.isdir(db_dir + '/' + name):
        print('Invalid database name')
        return
    if conn:
        stop_database(home_dir, db_dir, name, conn)
    os.system('rm -rf ' + db_dir + '/' + name)
    os.chdir(home_dir)