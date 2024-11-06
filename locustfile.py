from locust import HttpUser, task, between

class SpamDetectionUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def predict_spam(self):
        self.client.post("/predict", json={"text": "Free money awaits you!"})
