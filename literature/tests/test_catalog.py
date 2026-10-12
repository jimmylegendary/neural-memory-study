import importlib.util,json,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'tools/build_catalog.py'
spec=importlib.util.spec_from_file_location('catalog',P)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class CatalogueTests(unittest.TestCase):
    def test_dates_not_from_arxiv(self):
        self.assertEqual(m.date_value('2501.00663'),('','unknown'))
        self.assertEqual(m.date_value('2026-08'),('2026-08','month'))
        self.assertEqual(m.date_value('2026-08-32'),('','unknown'))
        self.assertEqual(m.date_value('2026.08'),('2026-08','month'))
        self.assertEqual(m.date_value('2025 (블로그)'),('2025','year'))
    def test_arxiv_base_versions(self):
        self.assertEqual(m.aid('https://arxiv.org/abs/2609.23466v2'),'2609.23466')
        self.assertNotEqual(m.aid('2607.23929'),m.aid('2607.27834'))
    def test_no_acronym_collision(self):
        rows=[{'title':'MemForest: Temporal Indexing','alias':'MemForest','scope':'repository_foundation','reported_arxiv':'2605.23986','observation_id':'a'}, {'title':'MemForest: EventTree','alias':'MemForest','scope':'task','reported_arxiv':'2609.07997','observation_id':'b'}]
        changes=m.resolve_rows(rows,{'aliases':{'MemForest':'2609.08273'}},[])
        self.assertEqual(rows[0]['canonical_arxiv'],'2605.23986')
        self.assertEqual(rows[1]['canonical_arxiv'],'2609.08273')
        self.assertEqual(len(changes),1)
    def test_nested_bib(self):
        block='title = {Fast {Weights} and {Memory}}, year={2021}, eprint={2102.11174}'
        self.assertEqual(m.bib_field(block,'title'),'Fast Weights and Memory')
        self.assertEqual(m.bib_field(block,'eprint'),'2102.11174')
    def test_parse_metadata_separates_dates(self):
        h=b'<meta name="citation_title" content="Titans: Learning to Memorize at Test Time"><meta name="citation_date" content="2024/12/31">[v1] Tue, 31 Dec 2024 12:00:00 UTC [v2] Mon, 06 Jan 2025 12:00:00 UTC'
        p=m.parse_abs(h,'2501.00663','2026-10-12')
        self.assertTrue(p['published'].startswith('2024-12-31'))
        self.assertTrue(p['updated'].startswith('2025-01-06'))
        self.assertEqual(p['latest_version'],2)
    def test_identity_matching(self):
        self.assertFalse(m.same_title('MemForest: Temporal Indexing','A Full-Stack Characterization of HBF'))
        self.assertTrue(m.same_title('RPMem','RPMem: Learning Long-Term Recurrent Parametric Memory'))
    def test_source_manifest_counts(self):
        lit=P.parents[1]
        imp=m.Importer(lit.parent,lit)
        imp.load_task()
        counts={k:v['input_count'] for k,v in imp.sources.items()}
        manifest=json.loads((lit/'inputs/source_manifest.json').read_text())
        for source in manifest['sources']:
            self.assertEqual(counts[source['source_id']],source['extracted_records'])
        self.assertEqual(counts['weekly-2026-10-12'],17)

if __name__=='__main__':
    unittest.main()
