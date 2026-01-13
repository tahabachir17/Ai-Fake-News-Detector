import pandas as pd
import sys
import os
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fake_news_detector.data.preprocessor import TextPreprocessor

def test_text_preprocessor():
    logging.info("Starting TextPreprocessor Verification")

    # Sample Data
    data = {
        'text': [
            "Check out this link: https://example.com!", 
            "<h1>BREAKING NEWS!!!!</h1>", 
            "This is a SIMPLE test with Common Stopwords like the and a.",
            "  Lot's   of   whitespace   here.  "
        ],
        'title': [
            "Link News",
            "HTML News",
            "Simple Title",
            "Whitespace Title"
        ]
    }
    df = pd.DataFrame(data)
    
    # Initialize Preprocessor
    preprocessor = TextPreprocessor(stopwords='english')
    
    # Fit (loads stopwords)
    preprocessor.fit(df)
    
    # Transform DataFrame
    logging.info("\n--- Testing DataFrame Transformation (Title + Text) ---")
    result_series = preprocessor.transform(df)
    
    print("\nOriginal DataFrame:")
    print(df)
    print("\nCleaned Series:")
    print(result_series)
    
    # Assertions
    assert isinstance(result_series, pd.Series), "Output should be a pandas Series"
    assert len(result_series) == len(df), "Output length should match input"
    
    # Check cleaning effects
    # 1. URL removal
    assert "https://example.com" not in result_series[0]
    # 2. HTML removal
    assert "<" not in result_series[1] and ">" not in result_series[1]
    # 3. Lowercasing
    assert "BREAKING" not in result_series[1] and "breaking" in result_series[1]
    # 4. Stopwords (check if 'this', 'is', 'a' are removed from "This is a SIMPLE test...")
    # 'simple' and 'test' should remain. 'this', 'is', 'a', 'the', 'and' are stopwords.
    assert "simple" in result_series[2]
    assert "this" not in result_series[2].split() # split to avoid substring match
    
    # Test List Input
    logging.info("\n--- Testing List Input ---")
    raw_list = ["  Random   Text  With 123  "]
    list_result = preprocessor.transform(raw_list)
    print("\nList Input Result:")
    print(list_result)
    assert list_result[0].strip() == "random text 123", f"Expected 'random text 123', got '{list_result[0]}'"

    logging.info("\nVerification Successful!")

if __name__ == "__main__":
    test_text_preprocessor()
