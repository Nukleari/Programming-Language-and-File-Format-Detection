from joblib import load

model = load('model.joblib')
vec = load('vec.joblib')

test_file = ''
count = 1
with open('example.py') as f:
    while True:
        line = f.readline()
        if not line:
            break
        test_file += line
        print(count, model.predict(vec.transform([test_file]))[0])
        count += 1