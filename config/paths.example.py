"""Exemplo centralizado de caminhos do Tech Challenge Fase 3.

Copie este arquivo para ``config/paths.py`` somente se o projeto precisar de
configuração local. O arquivo ``paths.py`` deve permanecer no ``.gitignore``.

O exemplo não contém credenciais. A autenticação e o acesso ao S3 devem ser
administrados pelo Unity Catalog e pelo External Volume do Databricks.
"""

from dataclasses import dataclass
import os
from pathlib import Path


DEFAULT_VOLUME_ROOT = Path(
    "/Volumes/workspace/default/vol_trio_drive"
)


@dataclass(frozen=True)
class ProjectPaths:
    """Caminhos de entrada da Fase 2 e resultados derivados da Fase 3."""

    volume_root: Path

    @property
    def fiap_root(self) -> Path:
        return self.volume_root / "projetos" / "fiap"

    @property
    def phase2_root(self) -> Path:
        return self.fiap_root / "tech_challenge_fase2"

    @property
    def phase3_root(self) -> Path:
        return self.fiap_root / "tech_challenge_fase3"

    @property
    def gold_input_dir(self) -> Path:
        """Única origem oficial mantida na estrutura da Fase 2."""

        return (
            self.phase2_root
            / "gold"
            / "alunos_base_enriquecida_finalizada"
        )

    @property
    def gold_input_file(self) -> Path:
        return self.gold_input_dir / "base_analitica_final.csv"

    @property
    def source_dir(self) -> Path:
        return self.phase3_root / "src"

    @property
    def modeling_dir(self) -> Path:
        return self.phase3_root / "modelagem"

    @property
    def modeling_results_dir(self) -> Path:
        return self.modeling_dir / "resultados"

    @property
    def artifacts_dir(self) -> Path:
        return self.phase3_root / "artifacts"

    @property
    def reports_dir(self) -> Path:
        return self.phase3_root / "reports"

    @property
    def modeling_reports_dir(self) -> Path:
        return self.reports_dir / "modelagem"

    @property
    def strategic_reports_dir(self) -> Path:
        return self.reports_dir / "aplicacao_estrategica"

    @property
    def visual_manifest_dir(self) -> Path:
        return self.reports_dir / "consolidacao_visual"

    @property
    def images_dir(self) -> Path:
        return self.phase3_root / "images"

    @property
    def final_images_dir(self) -> Path:
        return self.images_dir / "finais"


def load_paths() -> ProjectPaths:
    """Carrega a raiz por variável de ambiente ou usa o Volume oficial."""

    configured_root = os.getenv("TECH_CHALLENGE_VOLUME_ROOT")
    volume_root = Path(configured_root) if configured_root else DEFAULT_VOLUME_ROOT
    return ProjectPaths(volume_root=volume_root)


PATHS = load_paths()


if __name__ == "__main__":
    print("Entrada Gold:", PATHS.gold_input_file)
    print("Raiz da Fase 3:", PATHS.phase3_root)
    print("Código-fonte:", PATHS.source_dir)
    print("Artefatos:", PATHS.artifacts_dir)
    print("Relatórios:", PATHS.reports_dir)
    print("Imagens finais:", PATHS.final_images_dir)
