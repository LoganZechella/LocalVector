import os
import subprocess
import tempfile
import shutil
import logging

logger = logging.getLogger(__name__)

class GitRepositoryHandler:
    """Handler for cloning and managing Git repositories."""

    def __init__(self, repos_dir=None):
        """Initialize with optional directory for storing repositories."""
        self.repos_dir = repos_dir or tempfile.mkdtemp(prefix="localvector_repos_")
        logger.info(f"Initializing GitRepositoryHandler with repos directory: {self.repos_dir}")

    def clone_repository(self, repo_url, branch="main"):
        """Clone a Git repository to a local directory.

        Args:
            repo_url: URL of the repository to clone
            branch: Branch to checkout (default: main)

        Returns:
            Path to the cloned repository
        """
        # Extract repo name from URL
        repo_name = self._extract_repo_name(repo_url)

        # Create a directory for this repo
        repo_path = os.path.join(self.repos_dir, repo_name)

        # Check if already cloned
        if os.path.exists(repo_path):
            logger.info(f"Repository {repo_name} already exists at {repo_path}, pulling latest changes")
            self._update_repository(repo_path, branch)
        else:
            logger.info(f"Cloning repository {repo_url} to {repo_path}")
            self._clone_repository(repo_url, repo_path, branch)

        return repo_path

    def _extract_repo_name(self, repo_url):
        """Extract repository name from URL."""
        # Handle various URL formats
        if repo_url.endswith(".git"):
            repo_url = repo_url[:-4]

        # Extract the last part of the URL
        repo_name = repo_url.split("/")[-1]

        return repo_name

    def _clone_repository(self, repo_url, repo_path, branch):
        """Clone the repository to the specified path."""
        try:
            subprocess.run(
                ["git", "clone", "--branch", branch, "--single-branch", repo_url, repo_path],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e.stderr}")
            raise RuntimeError(f"Failed to clone repository: {e.stderr}")

    def _update_repository(self, repo_path, branch):
        """Update an existing repository to the latest commit."""
        try:
            # Change to the repository directory
            cwd = os.getcwd()
            os.chdir(repo_path)

            # Fetch and checkout the specified branch
            subprocess.run(["git", "fetch", "origin", branch], check=True, capture_output=True, text=True)
            subprocess.run(["git", "checkout", branch], check=True, capture_output=True, text=True)
            subprocess.run(["git", "pull", "origin", branch], check=True, capture_output=True, text=True)

            # Return to the original directory
            os.chdir(cwd)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to update repository: {e.stderr}")
            raise RuntimeError(f"Failed to update repository: {e.stderr}")

    def cleanup(self, keep_repos=False):
        """Clean up temporary directories.

        Args:
            keep_repos: If True, don't delete the repositories
        """
        if not keep_repos and os.path.exists(self.repos_dir):
            logger.info(f"Cleaning up repositories directory: {self.repos_dir}")
            shutil.rmtree(self.repos_dir) 