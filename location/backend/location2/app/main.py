from fastapi import FastAPI
app=FastAPI()

@app.get('/')
def hi():
    return {'msg':'location2'}
