"""Shared test fixtures for Finance RAG test suite."""

import pytest


@pytest.fixture
def sample_html_table() -> str:
    """A simple SEC-style HTML table for testing table extraction."""
    return """
    <table>
        <thead>
            <tr>
                <th></th>
                <th>2024</th>
                <th>2023</th>
                <th>2022</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Net sales</td>
                <td>$394,328</td>
                <td>$383,285</td>
                <td>$394,328</td>
            </tr>
            <tr>
                <td>Cost of sales</td>
                <td>$210,352</td>
                <td>$214,137</td>
                <td>$223,546</td>
            </tr>
            <tr>
                <td>Gross margin</td>
                <td>$183,976</td>
                <td>$169,148</td>
                <td>$170,782</td>
            </tr>
        </tbody>
    </table>
    """


@pytest.fixture
def sample_complex_html_table() -> str:
    """An HTML table with colspan/rowspan for testing fallback to HTML format."""
    return """
    <table>
        <tr>
            <th colspan="4">Consolidated Balance Sheet (in millions)</th>
        </tr>
        <tr>
            <th></th>
            <th>September 28, 2024</th>
            <th>September 30, 2023</th>
        </tr>
        <tr>
            <td colspan="3"><b>ASSETS</b></td>
        </tr>
        <tr>
            <td>Current assets:</td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>&nbsp;&nbsp;Cash and cash equivalents</td>
            <td>$29,943</td>
            <td>$29,965</td>
        </tr>
    </table>
    """


@pytest.fixture
def sample_xbrl_html() -> str:
    """HTML with XBRL tags for testing XBRL stripping and extraction."""
    return """
    <div>
        <ix:nonFraction contextRef="FD2024Q4YTD" decimals="-6"
            name="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
            unitRef="usd">394328000000</ix:nonFraction>
    </div>
    """


@pytest.fixture
def sample_prose_section() -> str:
    """A sample MD&A prose section for testing prose chunking."""
    return """
    <h2>Item 7. Management's Discussion and Analysis of Financial Condition
    and Results of Operations</h2>
    <p>The following discussion should be read in conjunction with the
    Consolidated Financial Statements and accompanying Notes thereto
    included in Item 8 of this Form 10-K. (See Note 2 for further detail.)</p>
    <p>Total net revenue increased 3 percent or $11.0 billion during 2024
    compared to 2023. The increase was primarily driven by higher sales
    of Services, partially offset by lower sales of iPhone.</p>
    <p>Products net revenue decreased 1 percent during 2024 compared to
    2023 due to lower net revenue from iPhone. Services net revenue
    increased 13 percent during 2024 compared to 2023 due to higher
    net revenue from advertising, the App Store and cloud services.
    (See Note 12 for segment breakdown.)</p>
    """


@pytest.fixture
def filing_metadata() -> dict:
    """Common filing metadata used across tests."""
    return {
        "company_ticker": "AAPL",
        "company_name": "Apple Inc.",
        "filing_type": "10-K",
        "fiscal_year": 2024,
        "filing_date": "2024-11-01",
    }
