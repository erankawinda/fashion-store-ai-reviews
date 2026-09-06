import unittest

import app as fashion_app


class FashionStoreSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fashion_app.load_model_and_data()
        fashion_app.app.config.update(TESTING=True)
        cls.client = fashion_app.app.test_client()
        cls.item_id = int(fashion_app.items_df.iloc[0]['Clothing ID'])

    def test_home_page_renders(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Fashion Store', response.data)

    def test_items_and_category_search(self):
        response = self.client.get('/api/items')
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertGreater(payload['count'], 0)
        self.assertLessEqual(payload['count'], 50)

        category = str(fashion_app.items_df.iloc[0]['Class Name'])
        search_response = self.client.get('/api/items', query_string={'search': category})
        search_payload = search_response.get_json()
        self.assertEqual(search_response.status_code, 200)
        self.assertGreater(search_payload['count'], 0)

        singular_response = self.client.get('/api/items', query_string={'search': 'dress'})
        self.assertEqual(singular_response.status_code, 200)
        self.assertGreater(singular_response.get_json()['count'], 0)

    def test_item_details_and_missing_item(self):
        response = self.client.get(f'/api/item/{self.item_id}')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['id'], self.item_id)
        self.assertEqual(self.client.get('/api/item/99999999').status_code, 404)

    def test_prediction_and_validation(self):
        response = self.client.post(
            '/api/predict',
            json={'review_title': 'Comfortable', 'review_text': 'A comfortable and useful item.'},
        )
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertIn(payload['recommendation'], (0, 1))
        self.assertGreaterEqual(payload['probability'], 0.0)
        self.assertLessEqual(payload['probability'], 1.0)

        self.assertEqual(self.client.post('/api/predict', json={}).status_code, 400)
        self.assertEqual(
            self.client.post('/api/predict', json={'review_text': ['not', 'text']}).status_code,
            400,
        )

    def test_review_round_trip_and_validation(self):
        response = self.client.post(
            '/api/reviews',
            json={
                'item_id': self.item_id,
                'title': 'Smoke-test review',
                'description': 'This review exists only in the test process.',
                'rating': 4,
                'recommendation': 1,
            },
        )
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        review_response = self.client.get(payload['review_url'])
        self.assertEqual(review_response.status_code, 200)
        self.assertEqual(review_response.get_json()['review_title'], 'Smoke-test review')

        invalid_rating = self.client.post(
            '/api/reviews',
            json={
                'item_id': self.item_id,
                'description': 'Invalid rating',
                'rating': 6,
                'recommendation': 1,
            },
        )
        self.assertEqual(invalid_rating.status_code, 400)


if __name__ == '__main__':
    unittest.main()
