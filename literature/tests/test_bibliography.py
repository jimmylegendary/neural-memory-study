import importlib.util,tempfile,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'tools/build_catalog.py'
spec=importlib.util.spec_from_file_location('catalog_bibliography',P)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class BibliographyTests(unittest.TestCase):
    def test_contiguous_bullets_are_distinct_papers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'literature').mkdir()
            (root/'bibliography.md').write_text('- Early, A. (1951). *A foundational result.* Journal.\n- Later, B. (2025). *Another result.* arXiv:2501.12345.\n',encoding='utf-8')
            imp=m.Importer(root,root/'literature');imp.load_bibliographies()
            self.assertEqual(len(imp.rows),2)
            self.assertEqual(imp.rows[0]['title'],'A foundational result')
            self.assertEqual(imp.rows[1]['title'],'Another result')
            self.assertEqual(imp.rows[1]['reported_arxiv'],'2501.12345')
    def test_multiline_title_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'literature').mkdir()
            (root/'bibliography.md').write_text('**Article**\n*A long title that wraps\nonto the next line.*\nPLoS Computational Biology, 2022-11-18.\n',encoding='utf-8')
            imp=m.Importer(root,root/'literature');imp.load_bibliographies()
            self.assertEqual(len(imp.rows),1)
            self.assertEqual(imp.rows[0]['title'],'A long title that wraps onto the next line')
    def test_short_method_name_requires_primary_evidence(self):
        row={'canonical_arxiv':'2610.11314','kind':'paper','title':'CogMem','alias':'CogMem','scope':'task','source_id':'weekly-test','observation_id':'test','source_date':'2026-10-12','reported_published':'','topics':'','note':''}
        cache={'2610.11314':{'status':'ok','title':'From Retrieval to Reconstruction','published':'2026-10-08','summary':'We introduce CogMem, a cognitive memory architecture.'}}
        p=m.build_publications([row],cache,[],{})[0]
        self.assertEqual(p['verification_status'],'primary_metadata_verified')
        cache['2610.11314']['summary']='An unrelated cost-effectiveness model.'
        p=m.build_publications([row],cache,[],{})[0]
        self.assertEqual(p['verification_status'],'identity_conflict')

if __name__=='__main__':unittest.main()
