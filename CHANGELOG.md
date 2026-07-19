# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0](https://github.com/simonjanssen/ganshot/compare/v0.1.1...v0.2.0) - 2026-07-19

### Added

- *(data)* HumanPose dataset ([#14](https://github.com/simonjanssen/ganshot/pull/14))

### Fixed

- *(release)* enable git-only mode ([#17](https://github.com/simonjanssen/ganshot/pull/17))
- fix typo ([#9](https://github.com/simonjanssen/ganshot/pull/9))

### Other

- *(deps)* cargo update ([#13](https://github.com/simonjanssen/ganshot/pull/13))
- release v0.1.1 ([#12](https://github.com/simonjanssen/ganshot/pull/12))
- Replace custom release workflow with release-plz ([#11](https://github.com/simonjanssen/ganshot/pull/11))
- Bump actions/checkout from 6 to 7 ([#10](https://github.com/simonjanssen/ganshot/pull/10))

## [0.1.1](https://github.com/simonjanssen/ganshot/releases/tag/v0.1.1) - 2026-07-16

### Added

- *(ci)* add release workflow ([#8](https://github.com/simonjanssen/ganshot/pull/8))

### Fixed

- fix typo ([#9](https://github.com/simonjanssen/ganshot/pull/9))
- fix dangling doc reference
- fix range

### Other

- Replace custom release workflow with release-plz ([#11](https://github.com/simonjanssen/ganshot/pull/11))
- Bump actions/checkout from 6 to 7 ([#10](https://github.com/simonjanssen/ganshot/pull/10))
- update licenses
- use install-action for additional cargo tools
- cargo update
- RUSTSEC-2026-0204
- regular audits
- add demo
- sample fixed + continuous
- custom plot locations
- cleanup
- bump anyhow
- no default burn features
- gate features
- optimize outline plots for points
- plot outlines over epochs
- trajectory plot for losses
- add gaussian example
- define a generic dataset and sampling logic + remove mnist example
- cargo audit findings
- replace hard-coded backend
- remove clone
- remove import
- gate import
- license config
- multi-platform device init
- ignore bincode/paste not being maintained
- go with wgpu backend / disable metal for now
- centralize in backend module
- update both models in every iteration
- don't lint ext deps
- add GAN model + training
- incl config file changes
- cleanup
- add inference example
- fmt
- add get started guide example
- default setup
- Initial commit
