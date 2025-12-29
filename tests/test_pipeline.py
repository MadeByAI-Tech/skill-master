from pipeline.phase1.main import extract

def test_processing_job_extract():
    assert extract("abcd") == "abcd-extracted"