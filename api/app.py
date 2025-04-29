import os
from threading import Thread
from flask import Flask, session, request, jsonify
from flask_cors import CORS
import datajoint as dj
import pymysql
import time
import json

# fix for wget
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

# import custom functions
from helpers.init import create_database, delete_database, start_database, stop_database
from helpers.pop import append_data
from helpers.query import saved_queries, add_query, delete_query
from helpers.query import query_levels, table_fields, create_query, generate_tree
from helpers.query import get_metadata_helper
from helpers.query import get_options, get_trace_binary, get_spikehist_binary
from helpers.query import add_tags, delete_tags
from helpers.query import push_tags, pull_tags, reset_tags

app = Flask(__name__)
CORS(app)

# immutable globals
home_dir: str = os.getcwd()
schema_path: str = './api/schema.py'
db_dir: str = os.path.abspath("../databases")#"/Users/samarjit/workspace/neuro/samarjit_dj_tool/datajoint/databases"#
download_dir: str = os.path.abspath("../downloads") # similar to above

# mutable globals (should be saved to a session)
mea_dir: str = None
db: dj.VirtualModule = None
username: str = "guest"
query: dj.expression.QueryExpression = None
exclude_levels: list = []

# progress tracking globals
add_data_started = False
add_data_error = None

# dj config
host_address, user, password = '127.0.0.1', 'root', 'simple'
dj.config["database.host"] = f"{host_address}"
dj.config["database.user"] = f"{user}"
dj.config["database.password"] = f"{password}"

print("Globals populated", flush=True)

### 1.1: Set database storage directory, initialize and connect to database

# dir: str -> None
@app.route('/init/set-database-directory', methods=['POST'])
def set_db_dir():
    global db_dir
    db_dir = request.json.get('dir')
    if db_dir and os.path.isdir(db_dir):
        return jsonify({"message": "Database directory set successfully!"}), 200
    else:
        return jsonify({"message": "Invalid directory path!"}), 400

# dir: str -> None
@app.route('/init/set-mea-directory', methods=['POST'])
def set_mea_dir():
    global mea_dir
    mea_dir = request.json.get('dir')
    if mea_dir and os.path.isdir(mea_dir):
        return jsonify({"message": "MEA data directory set successfully!"}), 200
    else:
        return jsonify({"message": "Invalid directory path!"}), 400

# None -> dir: str
@app.route('/init/get-database-directory', methods=['GET'])
def get_db_dir():
    return jsonify({"dir": f"{db_dir}"})

# None -> databases: list
@app.route('/init/list-databases', methods=['GET'])
def list_dbs():
    if db_dir:
        dbs = [f for f in os.listdir(db_dir) if os.path.isdir(os.path.join(db_dir, f))]
        return jsonify({"databases": dbs}), 200
    else:
        return jsonify({"message": "Database directory not set!"}), 400

# name: str -> None
@app.route('/init/create-database', methods=['POST'])
def create_db():
    db_name = request.json.get('name')
    if db_name and db_dir:
        try:
            create_database(home_dir, db_dir, db_name)
            return jsonify({"message": "Database created successfully!"}), 200
        except Exception as e:
            return jsonify({"message": f"Error creating database: {e}"}), 400
    else:
        return jsonify({"message": "Invalid database name!"}), 400
    
# name: str -> None
@app.route('/init/delete-database', methods=['POST'])
def delete_db():
    db_name = request.json.get('name')
    if db_name and db_dir:
        try:
            delete_database(home_dir, db_dir, db_name,
                            dj.conn() if hasattr(dj.conn, 'connection') else None)
            return jsonify({"message": "Database deleted successfully!"}), 200
        except Exception as e:
            return jsonify({"message": f"Error deleting database: {e}"}), 400
    else:
        return jsonify({"message": "Invalid database name!"}), 400

# name: str -> None
@app.route('/init/start-database', methods=['POST'])
def start_db():
    db_name = request.json.get('name')
    if db_name and db_dir:
        try:
            start_database(home_dir, db_dir, db_name)
            return jsonify({"message": "Database started successfully!"}), 200
        except Exception as e:
            return jsonify({"message": f"Error starting database: {e}"}), 400
    else:
        return jsonify({"message": "Invalid database name!"}), 400

# name: str -> None
@app.route('/init/stop-database', methods=['POST'])
def stop_db():
    global db
    db_name = request.json.get('name')
    if db_name and db_dir:
        try:
            stop_database(home_dir, db_dir, db_name,
                          dj.conn() if hasattr(dj.conn, 'connection') else None)
            return jsonify({"message": "Database stopped successfully!"}), 200
        except Exception as e:
            return jsonify({"message": f"Error stopping database: {e}"}), 400
    else:
        return jsonify({"message": "Invalid database name!"}),

# name: str -> None
@app.route('/init/connect-database', methods=['POST'])
def connect_db():
    global db
    db_name = request.json.get('name')
    if db_name and db_dir:
        try:
            for attempt in range(4):
                try:
                    if not dj.conn().is_connected:
                        dj.conn().connect()
                except pymysql.OperationalError as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(1)
            if not dj.conn().is_connected:
                dj.conn().connect()
            print('Connected' if dj.conn().is_connected else 'Failed to connect', flush=True)
            if 'schema' not in dj.list_schemas():
                print('Initializing schema')
                exec(open(schema_path).read())
            db = dj.VirtualModule('schema.py', 'schema')
            return jsonify({"message": "Connected to database successfully!"}), 200
        except Exception as e:
            print(f"Error connecting to database: {e}", flush=True)
            return jsonify({"message": f"Error connecting to database: {e}"}), 400
    else:
        return jsonify({"message": "Invalid database name!"}), 400

# None -> connected: bool
@app.route('/init/is-connected', methods=['GET'])
def is_connected():
    if db:
        return jsonify({"connected": True}), 200
    else:
        return jsonify({"connected": False}), 200

# 1.2: Setting user, should be a connection 'db' at this point. 

# user: str -> None
@app.route('/user/set-user', methods=['POST'])
def set_user():
    global username
    username = request.json.get('user')
    if username and '/' not in username and username != "user_not_set":
        return jsonify({"message": "User set successfully!"}), 200
    else:
        username = None
        return jsonify({"message": "Invalid user name!"}), 400

# None -> user: str
@app.route('/user/get-user', methods=['GET'])
def get_user():
    if username:
        return jsonify({"user": f"{username}"}), 200
    else:
        return jsonify({"user": "user_not_set"}), 200

# 1.3: once the user is set, we can start adding data.

# None -> empty: bool, num_experiments: int
@app.route('/pop/is-empty', methods=['GET'])
def is_empty():
    if db:
        num_experiments = len((db.Experiment & "id>=1").fetch())
        if num_experiments == 0:
            return jsonify({"empty": True, "num_experiments": num_experiments}), 200
        else:
            return jsonify({"empty": False, "num_experiments": num_experiments}), 200
    else:
        return jsonify({"message": "No database connection!"}), 400

def add_data_thread(data_dir, meta_dir, tags_dir, username, db):
    global add_data_started
    global add_data_error
    add_data_started = True
    try:
        append_data(data_dir, meta_dir, tags_dir, username, db)
    except Exception as e:
        print(f"Error adding to database: {e}", flush=True)
        add_data_started = False
    add_data_started = False

# data_dir: str, meta_dir: str, tags_dir: str -> None
@app.route('/pop/add-data', methods=['POST'])
def add_data():
    if db and username:
        if add_data_started:
            return jsonify({"message": "Data addition already in progress!"}), 400
        else:
            try:
                Thread(target=add_data_thread, args=(request.json.get('data_dir'), request.json.get('meta_dir'),
                                            request.json.get('tags_dir'), username, db)).start()
                return jsonify({"message": "Successfully started adding data! This can take a while."}), 200
            except Exception as e:
                print(f"Error adding to database: {e}", flush=True)
                return jsonify({"message": f"Error adding to database: {e}"}), 400
    else:
        return jsonify({"message": "Connect and sign in first!"}), 400

# None -> adding: bool
@app.route('/pop/is-adding', methods=['GET'])
def is_adding():
    if add_data_error:
        return jsonify({"message": add_data_error}), 400
    return jsonify({"adding": add_data_started}), 200

# None -> None
@app.route('/pop/clear', methods=['POST'])
def clear():
    if db and username:
        try:
            dj.config["safemode"] = False
            db.Experiment.delete()
            db.Protocol.delete()
            db.Tags.delete()
            dj.config["safemode"] = True
            return jsonify({"message": f"Database successfully cleared!"}), 200
        except:
            return jsonify({"message": "Error while clearing."}), 400
    else:
        return jsonify({"message": "Connect and sign in first!"}), 400
    
### 2.1: Querying: First we need methods to help create the query.

@app.route('/query/get-query-levels', methods=['GET'])
def get_query_levels():
    if db:
        return jsonify({"levels": query_levels()}), 200
    else:
        return jsonify({"message": "No database connection!"}), 400

# table: str -> fields: list
@app.route('/query/get-table-fields', methods=['POST'])
def get_table_fields():
    if db and username:
        return jsonify({"fields": table_fields(request.json.get('table_name'), username, db)}), 200
    else:
        return jsonify({"message": "No database connection!"}), 400

# All in one method (to replace the above two)
# None -> levels: list, fields: dict, tag_fields: list
@app.route('/query/get-levels-and-fields', methods=['GET'])
def get_levels_and_fields():
    if db and username:
        levels = query_levels()
        fields = {level: table_fields(level, username, db) for level in levels}
        tag_fields = table_fields('tags', username, db)
        return jsonify({"levels": levels, "fields": fields, "tag_fields": tag_fields}), 200
    else:
        return jsonify({"message": "No database connection!"}), 400
    
# methods to inject saved queries

# None -> queries: dict
@app.route('/query/get-saved-queries', methods=['GET'])
def get_saved_queries():
    if db and username:
        return jsonify({"queries": saved_queries(download_dir)}), 200
    else:
        return jsonify({"message": "No database connection!"}), 400
    
# query_name: str, query_obj: dict -> None
@app.route('/query/add-saved-query', methods=['POST'])
def add_saved_query():
    if db and username:
        try:
            add_query(request.json.get('query_name'), request.json.get('query_obj'), download_dir)
            return jsonify({"message": "Query saved successfully!"}), 200
        except Exception as e:
            return jsonify({"message": f"Error saving query: {e}"}), 400
    else:
        return jsonify({"message": "Connect and sign in first!"}), 400
    
# query_name: str -> None
@app.route('/query/delete-saved-query', methods=['POST'])
def delete_saved_query():
    if db and username:
        try:
            delete_query(request.json.get('query_name'), download_dir)
            return jsonify({"message": "Query deleted successfully!"}), 200
        except Exception as e:
            return jsonify({"message": f"Error deleting query: {e}"}), 400
    else:
        return jsonify({"message": "Connect and sign in first!"}), 400

# 2.2: Now we can actually execute the query!    

# query_obj: dict, exclude_levels: list -> results: list
@app.route('/query/execute-query', methods=['POST'])
def execute_query():
    if db and username:
        global query
        global exclude_levels
        # try:
        print("Querying", flush=True)
        query = create_query(request.json.get('query_obj'), username, db)
        print("Constructed query", flush=True)
        if query is not None:
            if len(query) > 0:
                exclude_levels = request.json.get('exclude_levels')
                tree = generate_tree(query, exclude_levels)
                print("Query executed", flush=True)
                return jsonify({"results": tree}), 200
            else:
                return jsonify({"message": f"{len(query)} results found!"}), 200
        # except Exception as e:
        #     return jsonify({"message": f"Error executing query: {e}"}), 400
    else:
        return jsonify({"message": "Connect and sign in first!"}), 400
    
# 3: Results methods: you can add your own visualizations here as well

def download_thread(query, bool_exclude_levels, bool_include_meta, filename):
    try:
        tree = generate_tree(query, 
                                 exclude_levels if bool_exclude_levels else [],
                                 bool_include_meta)
        print("Generated. Downloading to ", filename, flush=True)
        with open(filename, 'w') as f:
            # we must handle datetime objects
            f.write(json.dumps(tree, default=str))
        print("Downloaded", flush=True)
    except Exception as e:
        print(f"Error downloading results: {e}", flush=True)

# include_meta: bool, exclude_levels: bool -> None
@app.route('/results/download-results', methods=['POST'])
def download_results():
    if query:
        try:
            if not os.path.isdir(download_dir):
                os.mkdir(download_dir)
            filename = f"results_{time.strftime('%Y%m%d_%H%M%S')}.json"
            Thread(target=download_thread,
                   args=(query, request.json.get('exclude_levels'), 
                         request.json.get('include_meta'), f"{download_dir}/{filename}")).start()
            return jsonify({"message": 
                            f"Downloading to {filename}...\nThis can take a while, check progress in terminal"}), 200
        except Exception as e:
            return jsonify({"message": f"Error starting download: {e}"}), 400
    else:
        return jsonify({"message": "Run a query first!"}), 400

@app.route('/results/get-metadata', methods=['POST'])
def get_metadata():
    if db and username:
        try:
            metadata: dict = get_metadata_helper(request.json.get('level'), request.json.get('id'))
            if metadata is None:
                return jsonify({"message": "Metadata not found!"}), 400
            return jsonify({"metadata": metadata}), 200
        except Exception as e:
            return jsonify({"message": f"Error fetching metadata: {e}"}), 400
    else:
        return jsonify({"message": "Connect and sign in first!"}), 400

# id: int, experiment_id: int, level: str
# -> data: dict[<optgroup>: list[label: str, <...data>], ...]
@app.route('/results/get-visualization-data', methods=['POST'])
def get_visualization_data():
    if db and username:
        try:
            data: dict = get_options(request.json.get('level'), request.json.get('id'), request.json.get('experiment_id'))
            if data is None:
                return jsonify({"message": "No visualizations available yet!"}), 200
            return jsonify({"options": data}), 200
        except Exception as e:
            return jsonify({"message": f"Error fetching visualization: {e}"}), 400
    else:
        return jsonify({"message": "Connect and sign in first!"}), 400

# h5_file: str, h5_path: str -> image: bytes
@app.route('/results/get-visualization', methods=['POST'])
def get_visualization():
    if db and username:
        try:
            data = request.json.get('data')
            if data['vis_type'] == 'epoch-singlecell':
                image: bytes = get_trace_binary(data['h5_file'], data['h5_path'])
            elif data['vis_type'] == 'epoch_block-mea':
                image: bytes = get_spikehist_binary(data['data_path'])
            else:
                return jsonify({"message": "Visualization type not supported!"}), 400
            if image is None:
                return jsonify({"message": "No visualizations available for this option!"}), 200
            return jsonify({"image": image}), 200
        except Exception as e:
            return jsonify({"message": f"Error fetching visualization: {e}"}), 400
    else:
        return jsonify({"message": "Connect and sign in first!"}), 400

# epoch_id: int, experiment_id: int, level: str -> image: bytes
# @app.route('/results/get-visualization', methods=['POST'])
# def get_visualization():
#     if db and username:
#         if request.json.get('level') == 'epoch':
#             try:
#                 image: bytes = get_image_binary(request.json.get('id'), request.json.get('experiment_id'))
#                 if image is None:
#                     return jsonify({"message": "No visualizations available for this epoch!"}), 200
#                 return jsonify({"image": image}), 200
#             except Exception as e:
#                 return jsonify({"message": f"Error fetching visualization: {e}"}), 400
#         else:
#             return jsonify({"message": "No visualizations available yet!"}), 200
#     else:
#         return jsonify({"message": "Connect and sign in first!"}), 400

# ids: list ["experiment_id-level-id"], tag: str -> None
@app.route('/results/add-tags', methods=['POST'])
def bulk_add_tags():
    if db and username:
        try:
            ids = request.json.get('ids')
            tag = request.json.get('tag')
            add_tags(ids, tag)
            return jsonify({"message": "Tag added successfully!"}), 200
        except Exception as e:
            return jsonify({"message": f"Error adding tag: {e}"}), 400
    else:
        return jsonify({"message": "Connect and sign in first!"}), 400

# ids: list ["experiment_id-level-id"], tag: str -> None
@app.route('/results/delete-tags', methods=['POST'])
def bulk_delete_tags():
    if db and username:
        try:
            ids = request.json.get('ids')
            tag = request.json.get('tag')
            delete_tags(ids, tag)
            return jsonify({"message": "Tag deleted successfully!"}), 200
        except Exception as e:
            return jsonify({"message": f"Error deleting tag: {e}"}), 400
    else:
        return jsonify({"message": "Connect and sign in first!"}), 400

# experiment_ids: list -> None
@app.route('/results/push-tags', methods=['POST'])
def export_push_tags():
    if db and username:
        try:
            push_tags(request.json.get('experiment_ids'))
            return jsonify({"message": "Tags exported successfully!"}), 200
        except Exception as e:
            return jsonify({"message": f"Error exporting tags: {e}"}), 400
    else:
        return jsonify({"message": "Connect and sign in first!"}), 400
    
# experiment_ids: list -> None
@app.route('/results/pull-tags', methods=['POST'])
def import_pull_tags():
    if db and username:
        try:
            pull_tags(request.json.get('experiment_ids'))
            return jsonify({"message": "Tags imported successfully!"}), 200
        except Exception as e:
            return jsonify({"message": f"Error importing tags: {e}"}), 400
    else:
        return jsonify({"message": "Connect and sign in first!"}), 400
    
# experiment_ids: list -> None
@app.route('/results/reset-tags', methods=['POST'])
def import_reset_tags():
    if db and username:
        try:
            reset_tags(request.json.get('experiment_ids'))
            return jsonify({"message": "Tags reset successfully!"}), 200
        except Exception as e:
            return jsonify({"message": f"Error resetting tags: {e}"}), 400
    else:
        return jsonify({"message": "Connect and sign in first!"}), 400