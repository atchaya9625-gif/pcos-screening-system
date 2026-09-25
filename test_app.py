import io
import os
import unittest
from app import create_app
from app.models import db, PredictionHistory


class PCOSTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

    def test_01_landing_page(self):
        """Test GET / landing page loads successfully."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'PCOS', response.data)
        print("PASS: Landing page loads successfully.")

    def test_02_predict_page_get(self):
        """Test GET /predict renders form."""
        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'PCOS Risk Prediction', response.data)
        print("PASS: /predict GET renders form.")

    def test_03_predict_clinical_only(self):
        """Test POST /predict with clinical features only."""
        data = {
            'follicle_r': '14',
            'follicle_l': '12',
            'skin_darkening': '1',
            'hair_growth': '1',
            'weight_gain': '1',
            'cycle': 'I',
            'cycle_length': '38',
            'amh': '6.8',
            'prl': '22.5',
            'fsh_lh': '0.75',
            'fast_food': '1',
            'pimples': '1',
        }
        response = self.client.post('/predict', data=data)
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertIn('prediction', json_data)
        self.assertIn('risk_score', json_data)
        self.assertIn('shap_chart', json_data)
        self.assertTrue(json_data['shap_chart'].startswith('data:image/png;base64,'))
        self.assertIsNone(json_data['image_risk_score'])
        self.assertIsNone(json_data['gradcam_chart'])
        self.assertIn('clinical only', json_data['modality_used'])
        print(f"PASS: Clinical-only prediction returned risk: {json_data['risk_score']} ({json_data['prediction']}).")

    def test_04_predict_multimodal(self):
        """Test POST /predict with both clinical features and ultrasound scan."""
        test_img_path = os.path.join(os.path.dirname(__file__), 'data', 'test', 'infected', 'img_0_1033.jpg')
        self.assertTrue(os.path.exists(test_img_path), f"Test image not found at {test_img_path}")

        with open(test_img_path, 'rb') as img_f:
            img_bytes = img_f.read()

        data = {
            'follicle_r': '15',
            'follicle_l': '14',
            'skin_darkening': '1',
            'hair_growth': '1',
            'weight_gain': '1',
            'cycle': 'I',
            'cycle_length': '40',
            'amh': '7.2',
            'prl': '25.0',
            'fsh_lh': '0.6',
            'fast_food': '1',
            'pimples': '1',
            'ultrasound': (io.BytesIO(img_bytes), 'img_0_1033.jpg', 'image/jpeg'),
        }

        response = self.client.post('/predict', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertIn('prediction', json_data)
        self.assertIn('risk_score', json_data)
        self.assertIsNotNone(json_data['image_risk_score'])
        self.assertIsNotNone(json_data['gradcam_chart'])
        self.assertTrue(json_data['shap_chart'].startswith('data:image/png;base64,'))
        self.assertTrue(json_data['gradcam_chart'].startswith('data:image/png;base64,'))
        self.assertIn('multimodal', json_data['modality_used'])
        print(f"PASS: Multimodal prediction returned combined risk: {json_data['risk_score']}, clinical: {json_data['clinical_risk_score']}, image: {json_data['image_risk_score']}.")

    def test_05_predict_validation(self):
        """Test POST /predict input validation catches negative values."""
        data = {
            'follicle_r': '-5',
            'follicle_l': '10',
            'skin_darkening': '0',
            'hair_growth': '0',
            'weight_gain': '0',
            'cycle': 'R',
            'cycle_length': '28',
            'amh': '3.0',
            'prl': '15.0',
            'fsh_lh': '1.5',
            'fast_food': '0',
            'pimples': '0',
        }
        response = self.client.post('/predict', data=data)
        self.assertEqual(response.status_code, 400)
        json_data = response.get_json()
        self.assertIn('error', json_data)
        print(f"PASS: Input validation caught negative follicle count: {json_data['error']}.")

    def test_06_dashboard(self):
        """Test GET /dashboard renders recommendations."""
        response = self.client.get('/dashboard')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Personalized Lifestyle Dashboard', response.data)
        self.assertIn(b'Morning', response.data)
        self.assertIn(b'Afternoon', response.data)
        self.assertIn(b'Evening', response.data)
        self.assertIn(b'Night', response.data)
        print("PASS: Lifestyle dashboard renders with all 4 time slots.")

    def test_07_history(self):
        """Test GET /history renders history table with past predictions."""
        response = self.client.get('/history')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Prediction History', response.data)
        print("PASS: History page loads successfully.")


if __name__ == '__main__':
    unittest.main()
