import io
import json
import unittest
import zipfile
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
from backend.main import app, RUN_LOCK
from manifold_studio import Embedding


class ApiTests(unittest.TestCase):
    def setUp(self): self.client = TestClient(app)

    def test_frontend_and_catalog(self):
        self.assertIn('Give your embeddings', self.client.get('/').text)
        self.assertIn('smtl', self.client.get('/api/catalog').json()['losses'])
        self.assertEqual(self.client.get('/static/app.js').status_code, 200)

    def test_demo_product_plot_and_metrics(self):
        r = self.client.post('/api/run', json={'geometry':['sphere','torus']})
        self.assertEqual(r.status_code, 200, r.text)
        b = r.json(); self.assertEqual(b['report']['ambient_dim'], 7)
        self.assertTrue(b['plot']['data'])
        self.assertEqual(b['report']['rows'], 48)

    def test_vector_export_roundtrip(self):
        import tempfile
        from pathlib import Path
        x = np.random.default_rng(1).normal(size=(12, 8)).tolist()
        r = self.client.post('/api/export', json={'source':'vectors','vectors':x})
        self.assertEqual(r.status_code, 200, r.text[:200])
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            self.assertEqual(set(z.namelist()), {'embeddings.npz','projector.npz','metrics.json','coordinates.csv','explorer.html'})
            with tempfile.TemporaryDirectory() as t:
                p = Path(t)/'e.npz'; p.write_bytes(z.read('embeddings.npz'))
                np.testing.assert_allclose(Embedding.load(p).source, x)

    def test_validation_and_busy(self):
        for body in [{'geometry':['mobius']},{'steps':101},{'source':'vectors','vectors':[[1],[2]]},{'target':250}]:
            self.assertEqual(self.client.post('/api/run',json=body).status_code,422)
        with RUN_LOCK:
            self.assertEqual(self.client.post('/api/run',json={}).status_code,429)

    def test_text_path_without_network(self):
        text=['a sentence']*8
        with patch('manifold_studio.encoders.encode_sentences', return_value=np.random.default_rng(1).normal(size=(8,10))):
            r=self.client.post('/api/run',json={'source':'text','texts':text})
        self.assertEqual(r.status_code,200,r.text)

    def test_training_and_no_invented_triplets(self):
        r=self.client.post('/api/run',json={'mode':'train','loss':'smtl','steps':2})
        self.assertEqual(r.status_code,200,r.text)
        self.assertEqual(len(r.json()['report']['history']),3)
        x=np.random.default_rng(2).normal(size=(8,5)).tolist()
        r=self.client.post('/api/run',json={'mode':'train','source':'vectors','vectors':x})
        self.assertEqual(r.status_code,422)
        self.assertIn('triplet',r.json()['detail'])


if __name__=='__main__': unittest.main()
