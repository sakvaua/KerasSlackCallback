Simple Keras callback that pings your slack channel with training results

Usage:
1. Create a test Slack token. Legacy token allows you to do everything we need here.
https://get.slack.help/hc/en-us/articles/215770388-Create-and-regenerate-API-tokens
2. pip install slackclient
3. Configure the callback

#This is how you should store your tokens
#http://python-slackclient.readthedocs.io/en/latest/auth.html#handling-tokens
token='xoxp-yourtokengoeshere'#and not like this!!!
description='Mnist model MLP 512-RELU-DO0.2-512-RELU-DO0.2'#model descrition
message='Epoch {epoch:03d} loss:{val_loss:.4f} acc:{acc:.4f} val_loss:{val_loss:.4f} val_acc:{val_acc:.4f} '#format what to report 
slack=SlackCallback(token, channel='#general', model_description=description,mode='max', monitor='val_acc',message=message,best_only=True)

channel - which channel to post results to.
model_description - pretty self explanatory
mode 'min' or 'max' - lower of higher is better (accuracy - max, loss - min)
best_only - report only if there's an improvement


4. Use Callback
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),callbacks=[slack])


Dependencies:
Tensorflow 1.1+
Keras 2.x
slackclient
