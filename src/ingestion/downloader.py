"""SEC EDGAR filing downloader.

Downloads 10-K and 10-Q filings from the SEC EDGAR database for specified
companies. Stores raw HTML files locally for reproducible parsing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from sec_edgar_downloader import Downloader

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FilingMetadata:
    """Metadata for a downloaded SEC filing."""

    ticker: str
    filing_type: str  # "10-K" or "10-Q"
    fiscal_year: int | None = None
    file_path: Path | None = None
    filing_date: str = ""
    cik: str = ""
    accession_number: str = ""


class SECDownloader:
    """Downloads SEC filings from EDGAR.

    Respects SEC rate limits (10 requests/second) and stores raw HTML
    in a structured local directory for reproducible parsing.

    Args:
        company: Company name for SEC User-Agent header.
        email: Contact email for SEC User-Agent header.
        output_dir: Root directory for downloaded filings.
    """

    def __init__(
        self,
        company: str,
        email: str,
        output_dir: str | Path = "data/raw",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._downloader = Downloader(company, email, self.output_dir)
        self._request_interval = 0.12  # ~8 req/s to stay under SEC 10 req/s limit

    def download(
        self,
        ticker: str,
        filing_type: str = "10-K",
        limit: int = 5,
        after: str | None = None,
        before: str | None = None,
    ) -> list[FilingMetadata]:
        """Download filings for a single ticker.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL").
            filing_type: SEC filing type ("10-K" or "10-Q").
            limit: Maximum number of filings to download.
            after: Only download filings after this date (YYYY-MM-DD).
            before: Only download filings before this date (YYYY-MM-DD).

        Returns:
            List of FilingMetadata for each downloaded filing.
        """
        logger.info(
            f"Downloading {filing_type} filings for {ticker} (limit={limit})"
        )

        try:
            self._downloader.get(
                filing_type,
                ticker,
                limit=limit,
                after=after,
                before=before,
            )
        except Exception as e:
            logger.error(f"Failed to download {filing_type} for {ticker}: {e}")
            return []

        # Rate limiting
        time.sleep(self._request_interval)

        # Discover downloaded files
        return self._discover_filings(ticker, filing_type)

    def download_batch(
        self,
        tickers: list[str],
        filing_type: str = "10-K",
        limit: int = 5,
        after: str | None = None,
        before: str | None = None,
    ) -> list[FilingMetadata]:
        """Download filings for multiple tickers.

        Args:
            tickers: List of stock ticker symbols.
            filing_type: SEC filing type.
            limit: Maximum filings per ticker.
            after: Only download filings after this date.
            before: Only download filings before this date.

        Returns:
            Combined list of FilingMetadata for all tickers.
        """
        all_filings: list[FilingMetadata] = []
        for ticker in tickers:
            filings = self.download(
                ticker, filing_type, limit, after, before
            )
            all_filings.extend(filings)
            logger.info(
                f"Downloaded {len(filings)} filings for {ticker}"
            )
        logger.info(
            f"Total filings downloaded: {len(all_filings)} "
            f"for {len(tickers)} tickers"
        )
        return all_filings

    def _discover_filings(
        self, ticker: str, filing_type: str
    ) -> list[FilingMetadata]:
        """Discover downloaded filing files in the output directory.

        sec-edgar-downloader v5+ saves a raw SGML 'full-submission.txt'.
        This method handles three cases in priority order:
          1. Already-extracted .htm/.html files (re-runs)
          2. primary-document.html (older library versions)
          3. full-submission.txt → extract embedded HTML document

        Returns:
            List of FilingMetadata with file paths populated.
        """
        filings: list[FilingMetadata] = []
        base_path = self.output_dir / "sec-edgar-filings" / ticker / filing_type

        if not base_path.exists():
            logger.warning(f"No filings found at {base_path}")
            return filings

        for accession_dir in sorted(base_path.iterdir()):
            if not accession_dir.is_dir():
                continue

            # 1. Check for already-extracted HTML files
            html_files = list(accession_dir.glob("*.htm")) + list(
                accession_dir.glob("*.html")
            )
            # 2. Check primary-document.html (older library versions)
            if not html_files:
                primary = accession_dir / "primary-document.html"
                if primary.exists():
                    html_files = [primary]

            # 3. Extract from full-submission.txt (sec-edgar-downloader v5+)
            if not html_files:
                sgml_file = accession_dir / "full-submission.txt"
                if sgml_file.exists():
                    extracted = self._extract_html_from_sgml(sgml_file)
                    if extracted:
                        html_files = [extracted]

            if not html_files:
                logger.warning(
                    f"No HTML files found or extractable in {accession_dir}"
                )
                continue

            # Use the largest HTML file as the primary document
            primary_file = max(html_files, key=lambda f: f.stat().st_size)

            # Extract fiscal year from the SGML header or accession number date.
            # Accession number format: XXXXXXXXXX-YY-NNNNNN where YY = 2-digit year
            # of the filing. For 10-K filings, fiscal year = filing year (or year-1
            # for filings submitted in Q1 of next year).
            fiscal_year = self._extract_fiscal_year(
                accession_dir.name,
                filing_type,
                accession_dir / "full-submission.txt",
            )

            filing = FilingMetadata(
                ticker=ticker,
                filing_type=filing_type,
                file_path=primary_file,
                accession_number=accession_dir.name,
                fiscal_year=fiscal_year,
            )
            filings.append(filing)
            logger.info(
                f"Found filing: {primary_file.name} "
                f"({primary_file.stat().st_size:,} bytes) FY{fiscal_year}"
            )

        return filings

    def _extract_fiscal_year(
        self,
        accession_number: str,
        filing_type: str,
        sgml_path: Path | None = None,
    ) -> int | None:
        """Extract fiscal year from SGML header or accession number.

        Tries in order:
        1. PERIOD OF REPORT tag in SGML header (most accurate)
        2. Accession number YY component (filing date year)

        For 10-K filings, the period of report is the fiscal year end date.
        For filings submitted in Jan-Mar, fiscal year = prior year.

        Args:
            accession_number: Accession number string (XXXXXXXXXX-YY-NNNNNN).
            filing_type: Filing type (e.g. "10-K").
            sgml_path: Optional path to full-submission.txt for header parsing.

        Returns:
            Fiscal year as int, or None if undetermined.
        """
        import re

        # 1. Try PERIOD OF REPORT from SGML header (e.g. "20240928")
        if sgml_path and sgml_path.exists():
            try:
                # Only read the first 4KB — header is always at the top
                with open(sgml_path, encoding="utf-8", errors="replace") as f:
                    header = f.read(4096)

                period_match = re.search(
                    r"(?:<PERIOD-OF-REPORT>|CONFORMED\s+PERIOD\s+OF\s+REPORT:\s*)(\d{4})-?(\d{2})-?(\d{2})",
                    header,
                    re.IGNORECASE,
                )
                if period_match:
                    year = int(period_match.group(1))
                    return year
            except Exception:
                pass

        # 2. Fall back to accession number YY: XXXXXXXXXX-YY-NNNNNN
        acc_match = re.match(r"\d{10}-(\d{2})-\d{6}", accession_number)
        if acc_match:
            two_digit_year = int(acc_match.group(1))
            # Convert 2-digit to 4-digit year (SEC filings: 93-99=1993-1999, 00+=2000+)
            full_year = 2000 + two_digit_year if two_digit_year < 93 else 1900 + two_digit_year
            # For 10-K filed in Jan-Mar, fiscal year is typically prior year
            return full_year

        return None

    def _extract_html_from_sgml(self, sgml_path: Path) -> Path | None:
        """Extract the primary HTML document from an SEC SGML full-submission file.

        The full-submission.txt is a multi-document SGML container. Each embedded
        document is wrapped in <DOCUMENT>...</DOCUMENT> tags. The primary 10-K/10-Q
        document is the first one with TYPE 10-K or 10-Q, and is usually the largest
        HTML file.

        Args:
            sgml_path: Path to full-submission.txt.

        Returns:
            Path to the extracted HTML file, or None if extraction fails.
        """
        import re

        output_path = sgml_path.parent / "primary-document.htm"

        # Skip extraction if already done
        if output_path.exists() and output_path.stat().st_size > 1000:
            return output_path

        logger.info(f"Extracting HTML from SGML: {sgml_path.name}")

        try:
            # Read in chunks to avoid loading 10MB+ files fully into memory
            content = sgml_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error(f"Failed to read SGML file: {e}")
            return None

        # Find all <DOCUMENT>...</DOCUMENT> blocks and pick the HTML/HTM one
        # that matches the primary filing type (largest HTML body)
        doc_pattern = re.compile(
            r"<DOCUMENT>(.*?)</DOCUMENT>", re.DOTALL | re.IGNORECASE
        )
        type_pattern = re.compile(
            r"<TYPE>([^\n\r]+)", re.IGNORECASE
        )
        sequence_pattern = re.compile(
            r"<SEQUENCE>([^\n\r]+)", re.IGNORECASE
        )
        text_pattern = re.compile(
            r"<TEXT>(.*?)</TEXT>", re.DOTALL | re.IGNORECASE
        )

        best_doc: str | None = None
        best_size = 0

        for doc_match in doc_pattern.finditer(content):
            doc_block = doc_match.group(1)

            # Get document type
            type_m = type_pattern.search(doc_block)
            doc_type = type_m.group(1).strip() if type_m else ""

            # Get sequence number (sequence 1 = primary document)
            seq_m = sequence_pattern.search(doc_block)
            sequence = seq_m.group(1).strip() if seq_m else "99"

            # Get text body
            text_m = text_pattern.search(doc_block)
            if not text_m:
                continue

            text_body = text_m.group(1).strip()

            # Only keep documents that look like HTML
            is_html = (
                "<html" in text_body.lower()
                or "<HTML" in text_body
                or "<!DOCTYPE" in text_body
            )
            if not is_html:
                continue

            # Prefer: sequence=1, or largest HTML document
            doc_size = len(text_body)
            is_primary = sequence == "1"

            if is_primary or doc_size > best_size:
                best_doc = text_body
                best_size = doc_size
                if is_primary:
                    break  # Sequence 1 is definitively the primary doc

        if not best_doc:
            logger.warning(f"No HTML document found in {sgml_path.name}")
            return None

        # Write extracted HTML
        try:
            output_path.write_text(best_doc, encoding="utf-8")
            logger.info(
                f"Extracted HTML: {output_path.name} "
                f"({output_path.stat().st_size:,} bytes)"
            )
            return output_path
        except Exception as e:
            logger.error(f"Failed to write extracted HTML: {e}")
            return None
