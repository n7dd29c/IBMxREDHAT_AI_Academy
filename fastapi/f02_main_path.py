from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message" : "Hello, FastAPI"}

@app.get("/item/{item_id}")
def read_item(item_id):
    return {"item_id":item_id}

@app.get("/items/")
def read_itmes(skip=0, limit=10):           # default 0, 10 으로 지정
    return {'skip':skip, 'limit':limit}