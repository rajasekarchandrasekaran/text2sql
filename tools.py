from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.tools.sql_database.tool import ListSQLDatabaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
#llm
llm = ChatGoogleGenerativeAI(
    model = "gemini-1.5-flash",
    temperature=0
)

#db
db=SQLDatabase.from_uri("sqlite:///chinook.db")
# print(db.dialect)
# print(db.get_usable_table_names())

#tools
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
